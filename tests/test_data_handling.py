import unittest
from data_handling import *


class TestDataHandling(unittest.TestCase):
    def test_parse_gnashy(self):
        entry = GeneEntry()
        entry.parse_gnashy_entry('YAL068C	PAU8	1	2169	1807	1.5594	4	7.3320e-02	1.5594	0	1.0000e+00'.split('\t'))
        self.assertEqual(entry.NAME, 'PAU8')
        self.assertEqual(entry.CHROM, '1')
        self.assertEqual(entry.START, 2169)
        self.assertEqual(entry.END, 1807)
        self.assertAlmostEqual(entry.PROMEXP, 1.5594, places=4)
        self.assertAlmostEqual(entry.PROMCNT, 4.0, places=5)

    def test_read_diff_expr(self):
        entry = GeneEntry()
        self.assertTrue(entry.parse_gnashy_entry('YAL068C	PAU8	1	2169	1807	1.5594	4	7.3320e-02	1.5594	0	0.5'.split('\t'), pval=0.5))
        self.assertFalse(entry.parse_gnashy_entry('YAL068C	PAU8	1	2169	1807	1.5594	4	7.3320e-02	1.5594	0	0.5'.split('\t'), pval=0.005))
        entry_dict = {'PAU8': entry}
        read_diff_expr(['2.33396e+06'], ['YAL068C	PAU8'], entry_dict, 'test')
        self.assertEqual(entry_dict['PAU8'].DIFEXPRLVLS['test'], 2.33396E6)


if __name__ == '__main__':
    unittest.main()
