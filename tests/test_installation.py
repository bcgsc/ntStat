import subprocess
import unittest


class TestEntryPoint(unittest.TestCase):

    def test_main_entry_point(self):
        proc = subprocess.run(["ntstat", "--version"], stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)

    def test_count_entry_point(self):
        proc = subprocess.run(["ntstat", "count", "--help"], stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)
    
    def test_filter_entry_point(self):
        proc = subprocess.run(["ntstat", "filter", "--help"], stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)

    def test_hist_entry_point(self):
        proc = subprocess.run(["ntstat", "hist", "--help"], stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)

    def test_query_entry_point(self):
        proc = subprocess.run(["ntstat", "query", "--help"], stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)
