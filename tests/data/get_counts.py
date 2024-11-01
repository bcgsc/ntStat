import sys

import btllib
import numpy as np

outs = dict()

k = int(sys.argv[1])
for file in sys.argv[2:-1]:
    sr = btllib.SeqReader(file, btllib.SeqReaderFlag.SHORT_MODE)
    for record in sr:
        seq = record.seq
        kmer_set = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            kmer = min(kmer, btllib.get_reverse_complement(kmer))
            if kmer not in outs:
                outs[kmer] = {"count": 0, "depth": 0}
            outs[kmer]["count"] += 1
            if kmer not in kmer_set:
                outs[kmer]["depth"] += 1
            kmer_set.add(kmer)
kmers, counts, depths = [], [], []
for k, v in outs.items():
    kmers.append(k)
    counts.append(v["count"])
    depths.append(v["depth"])
hashes = np.array(kmers)
counts = np.array(counts, dtype=np.uint16)
depths = np.array(depths, dtype=np.uint16)
np.savez_compressed(sys.argv[-1], kmers=kmers, counts=counts, depths=depths)
