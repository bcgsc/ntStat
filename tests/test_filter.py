import logging
import os
import subprocess
import tempfile
import unittest

import btllib
import numpy as np


class TestFilterModule(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("TestFilterModule")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(asctime)s | %(name)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        hist_file = os.path.join(data_dir, "reads_k30.hist")
        reads = os.path.join(data_dir, "reads.fa.gz")
        self.cmd = ["ntstat", "filter", "-k", "30", "-f", hist_file, reads]
        npz_file = np.load(os.path.join(data_dir, "reads_kmers.npz"))
        self.kmers = npz_file["kmers"]
        self.counts = npz_file["counts"]

    def _test(self, min_count: int, max_count: int, out_counts: bool) -> bool:
        log_prefix = f"cmin={min_count}, cmax={max_count}"
        out_file = tempfile.NamedTemporaryFile()
        cmd = self.cmd[:-1] + ["-cmin", str(min_count), "-cmax", str(max_count)]
        cmd += ["--counts"] if out_counts else []
        cmd += ["-o", out_file.name] + [self.cmd[-1]]
        self.logger.info(f"running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(os.path.isfile(out_file.name))
        if out_counts:
            out = btllib.KmerCountingBloomFilter8(out_file.name)
        else:
            out = btllib.KmerBloomFilter(out_file.name)
        self.logger.info(f"{log_prefix}: fpr = {out.get_fpr()}")
        self.assertLessEqual(out.get_fpr(), 2e-4)
        num_err = 0
        for kmer, count in zip(self.kmers, self.counts):
            expected = min_count <= count <= max_count
            if out_counts:
                true_count = min(count, 255) if expected else 0
                cbf_count = out.contains(kmer)
                if cbf_count > 0 and min_count == 2:
                    cbf_count += 1
                num_err += 1 if cbf_count != true_count else 0
            else:
                exists = out.contains(kmer) != 0
                num_err += 1 if exists != expected else 0
        err_rate = num_err / len(self.kmers)
        self.logger.info(f"{log_prefix}: {num_err} errors ({err_rate})")
        self.assertLessEqual(err_rate, 0.04)
        out_file.close()

    def test_cbf_min1_max255(self):
        self._test(1, 255, True)

    def test_cbf_min2_max255(self):
        self._test(2, 255, True)

    def test_cbf_min3_max255(self):
        self._test(3, 255, True)

    def test_cbf_min1_max20(self):
        self._test(1, 20, True)

    def test_cbf_min3_max20(self):
        self._test(3, 20, True)

    def test_cbf_min1_max3(self):
        self._test(1, 3, True)

    def test_bf_min1_max255(self):
        self._test(1, 255, False)

    def test_bf_min2_max255(self):
        self._test(2, 255, False)

    def test_bf_min3_max255(self):
        self._test(3, 255, False)
