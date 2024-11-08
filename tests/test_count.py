import logging
import os
import subprocess
import tempfile
import unittest

import btllib
import numpy as np


class TestCountModule(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("TestCountModule")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(asctime)s | %(name)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def test_counts(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        # run ntstat's count module
        hist_file = os.path.join(data_dir, "reads_k30.hist")
        reads = os.path.join(data_dir, "reads.fa.gz")
        out_dir = tempfile.TemporaryDirectory()
        out_prefix = os.path.join(out_dir.name, "out")
        cmd = ["ntstat", "count", "-k", "30", "-f", hist_file, "-o", out_prefix, reads]
        self.logger.info(f"running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)
        # check outputs
        counts_cbf_path = out_prefix + "counts.cbf"
        depths_cbf_path = out_prefix + "depths.cbf"
        self.logger.info(f"looking for {counts_cbf_path}")
        self.assertTrue(os.path.isfile(counts_cbf_path))
        self.logger.info(f"looking for {depths_cbf_path}")
        self.assertTrue(os.path.isfile(depths_cbf_path))
        # check cbf contents
        npz_path = os.path.join(data_dir, "reads_kmers.npz")
        self.logger.info(f"reading ground truths")
        npz_file = np.load(npz_path)
        kmers = npz_file["kmers"]
        counts = npz_file["counts"]
        depths = npz_file["depths"]
        self.logger.info(f"loaded {len(kmers)} kmers")
        counts_cbf = btllib.KmerCountingBloomFilter8(counts_cbf_path)
        depths_cbf = btllib.KmerCountingBloomFilter8(depths_cbf_path)
        self.logger.info(f"cbf size (bytes) = {counts_cbf.get_bytes()}")
        self.logger.info(f"counts fpr = {counts_cbf.get_fpr()}")
        self.logger.info(f"depths fpr = {depths_cbf.get_fpr()}")
        self.assertLess(counts_cbf.get_fpr(), 0.04)
        self.assertLess(depths_cbf.get_fpr(), 0.04)
        num_err = 0
        for kmer, count, depth in zip(kmers, counts, depths):
            count_err = counts_cbf.contains(kmer) != min(count, 255)
            depth_err = depths_cbf.contains(kmer) != min(depth, 255)
            num_err += 1 if count_err or depth_err else 0
        err_rate = num_err / len(kmers)
        self.logger.info(f"{num_err} errors ({err_rate})")
        self.assertLess(err_rate, 0.002)
        # remove temporary files
        out_dir.cleanup()
