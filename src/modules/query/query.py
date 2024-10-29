import argparse
import os
import sys

import btllib


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser("query")
    parser.add_argument("-b", help="path to BF/CBF file", required=True)
    parser.add_argument("-s", help="path to spaced seeds file")
    parser.add_argument("-o", help="path to output TSV file", required=True)
    parser.add_argument("data", help="path to query data", nargs="+")
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    args = parse_args(argv)
    if btllib.KmerBloomFilter.is_bloom_file(args.b):
        print("[-b] loading btllib::KmerBloomFilter... ", end="", flush=True)
        bf = btllib.KmerBloomFilter(args.b)
        hash_fn = btllib.NtHash
        hash_fn_args = (bf.get_hash_num(), bf.get_k())
    elif btllib.KmerCountingBloomFilter8.is_bloom_file(args.b):
        print("[-b] loading btllib::KmerCountingBloomFilter8... ", end="", flush=True)
        bf = btllib.KmerCountingBloomFilter8(args.b)
        hash_fn = btllib.NtHash
        hash_fn_args = (bf.get_hash_num(), bf.get_k())
    elif btllib.SeedBloomFilter.is_bloom_file(args.b):
        print("[-b] loading btllib::SeedBloomFilter... ", end="", flush=True)
        bf = btllib.SeedBloomFilter(args.b)
        hash_fn = btllib.SeedNtHash
        hash_fn_args = (bf.get_seeds(), bf.get_hash_num(), bf.get_k())
    else:
        print("invalid bloom filter file", file=sys.stderr)
        return 1
    print("done")
    all_kmers = set()
    out_file = open(args.o, "w")
    for file in args.data:
        print(f"querying {file}... ", end="", flush=True)
        seq_reader = btllib.SeqReader(file, btllib.SeqReaderFlag.LONG_MODE)
        for record in seq_reader:
            hash_iter = hash_fn(record.seq, *hash_fn_args)
            while hash_iter.roll():
                pos = hash_iter.get_pos()
                kmer = record.seq[pos : pos + hash_iter.get_k()]
                if kmer not in all_kmers:
                    all_kmers.add(kmer)
                    value = bf.contains(hash_iter.hashes())
                    out_file.write(f"{kmer}\t{value}{os.linesep}")
        print("done")
    return 0
