import argparse
import itertools
import multiprocessing
import os
import re
import random

from pathlib import Path
from typing import List

import ftfy
import tensorflow as tf
from lm_dataformat import Reader
from transformers import GPT2Tokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="""
    Converts a text dataset into the training data format expected by the model.

    Adapted from the script create_tfrecords.py in the gpt-neo repo.

    - Your text dataset:
        - can be provided as .txt files, or as an archive (.tar.gz, .xz, jsonl.zst).
        - can be one file or multiple
            - using a single large file may use too much memory and crash - if this occurs, split the file up into a few files
        - the model's end-of-text separator is added between the contents of each file
        - if the string '<|endoftext|>' appears inside a file, it is treated as the model's end-of-text separator (not the actual string '<|endoftext|>')
            - this behavior can be disabled with --treat-eot-as-text

    This script creates a single .tfrecords file as output
        - Why: the model's data loader ignores "trailing" data (< 1 batch) at the end of a .tfrecords file
            - this causes data loss if you have many .tfrecords files
        - This is probably not appropriate for very large datasets
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an input file, or a directory that contains the input files.",
    )
    parser.add_argument("name", type=str,
                        help="Name of output file will be {name}_{seqnum}.tfrecords, where seqnum is total sequence count")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory (default: current directory)")

    cleaning_args = parser.add_argument_group('data cleaning arguments')

    cleaning_args.add_argument(
        "--normalize-with-ftfy", action="store_true", help="Normalize text with ftfy")
    cleaning_args.add_argument("--normalize-with-wikitext-detokenize",
                               action="store_true", help="Use wikitext detokenizer")
    minu_help = "Exclude repetitive documents made up of < MIN_UNIQUE_TOKENS unique tokens. These can produce large gradients."
    minu_help += " Set <= 0 to disable. If enabled, 200 is a good default value. (Default: 0)"
    cleaning_args.add_argument("--min-unique-tokens", type=int, default=0,
                               help=minu_help)

    # Args used to create conv-dataset ['../input', 'conv', '--output-dir', '../output', '--normalize-with-ftfy', '--normalize-with-wikitext-detokenize']
    args = parser.parse_args()

    # convert input_path to pathy
    args.input_path = Path(args.input_path)

    return args


def get_files(input_path: Path) -> List[str]:
    supported_file_types = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]
    if input_path.is_dir():
        # get all files with supported file types
        files = [list(Path(input_path).glob(f"*{ft}"))
                 for ft in supported_file_types]
        # flatten list
        files = [f for sublist in files for f in sublist]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert any(
            str(input_path).endswith(f_type) for f_type in supported_file_types
        ), f"Input file type must be one of: {supported_file_types}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path=}")

    return [str(f) for f in files]


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # text artifacts
    string = string.replace(" <|", "<|")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    """
    writes data to tfrecord file
    """
    feature = {
        "text": _int64_feature(data)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_tfrecord(sequences, fp, pbar):
    with tf.io.TFRecordWriter(fp) as writer:
        for sequence in sequences:
            write_to_file(writer, sequence)
            with lock:
                pbar.update()


def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i + n] for i in range(0, len(l), n)]


def enforce_min_unique(seqs, min_unique_tokens, enc, verbose=False):
    for seq in tqdm(seqs, mininterval=1, smoothing=0, desc="enforce_min_unique_tokens"):
        if len(set(seq)) >= min_unique_tokens:
            yield seq
        elif verbose:
            text = enc.decode(seq)
            #print(f"excluding with {len(set(seq))} unique tokens:\n\n{repr(text)}\n\n")


def eot_splitting_generator(string_iterable, encoder):
    """
    Given strings, splits them internally on <|endoftext|> and yields (generally more) strings
    """
    for doc in string_iterable:
        for d in doc.split(encoder.eos_token):
            if len(d) > 0:
                yield d


def prep_and_tokenize_generator(string_iterable, encoder, normalize_with_ftfy, normalize_with_wikitext_detokenize):
    """
    Given strings, does data cleaning / tokenization and yields arrays of tokens
    """
    for doc in string_iterable:
        if normalize_with_ftfy:  # fix text with ftfy if specified
            doc = ftfy.fix_text(doc, normalization='NFKC')
        if normalize_with_wikitext_detokenize:
            doc = wikitext_detokenizer(doc)
        tokens = encoder.encode(doc) + [encoder.eos_token_id]
        yield tokens


def file_to_tokenized_docs_generator(file_path, encoder, args):
    """
    Given a file path, reads the file and tokenizes the contents

    Yields token arrays of arbitrary, unequal length
    """
    reader = Reader(file_path)
    string_iterable = reader.stream_data(threaded=False)
    string_iterable = eot_splitting_generator(string_iterable, encoder)

    token_list_gen = prep_and_tokenize_generator(string_iterable,
                                                 encoder,
                                                 normalize_with_ftfy=args.normalize_with_ftfy,
                                                 normalize_with_wikitext_detokenize=args.normalize_with_wikitext_detokenize
                                                 )
    return token_list_gen


def arrays_to_sequences(token_list_iterable, sequence_length=2049):
    """
    Given token arrays of arbitrary lengths, concats/splits them into arrays of equal length

    Returns equal-length token arrays, followed by a a final array of trailing tokens (which may be shorter)
    """
    accum = []
    for l in token_list_iterable:
        accum.extend(l)

        if len(accum) > sequence_length:
            chunks = split_list(accum, sequence_length)
            yield from chunks[:-1]
            accum = chunks[-1]

    if len(accum) > 0:
        yield accum


def chunk_and_finalize(arrays, encoder):
    sequences = arrays_to_sequences(arrays)

    #if args.min_unique_tokens > 0:
    #    full_seqs = list(enforce_min_unique(full_seqs, args.min_unique_tokens, encoder, args.verbose))

    for sequence in sequences:
        if len(sequence) == 2049:
            yield sequence


def create_tfrecord(args):
    file, args, num = args
    # disables a misleading warning
    GPT2Tokenizer.max_model_input_sizes['gpt2'] = 1e20
    encoder = GPT2Tokenizer.from_pretrained('gpt2')
    encoder.add_tokens(['<|endofturn|>'])

    #random.seed(args.seed)

    fp = os.path.join(
        args.output_dir, f"{args.name}_{num}.tfrecords")
    sequences = chunk_and_finalize(
        file_to_tokenized_docs_generator(file, encoder, args), encoder)
    pbar = tqdm(sequences, position=num+1,
                leave=False, desc=f"processing file {num}")
    write_tfrecord(sequences, fp, pbar)


def init(l, s):
    global lock
    global seq_count
    lock = l


def create_tfrecords(files, args):
    results = []
    l = multiprocessing.Lock()
    s = 0
    with multiprocessing.Pool(processes=1, initializer=init, initargs=(l,s)) as pool:
        for result in tqdm(
            pool.imap_unordered(
                create_tfrecord, zip(
                    files, itertools.repeat(args), range(len(files)))
            ), total=len(files),
                position=0,
                leave=None,
                desc='Total'):
            results.append(result)

    return results


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    files = get_files(args.input_path)
    print(f"Creating TFRecords from files: {files}")

    results = create_tfrecords(files, args)
