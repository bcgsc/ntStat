import subprocess
import unittest


class TestEntryPoint(unittest.TestCase):

    def test_entry_point(self):
        proc = subprocess.run(["ntstat", "--version"], stdout=subprocess.DEVNULL)
        self.assertEqual(proc.returncode, 0)
