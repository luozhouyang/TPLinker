import unittest

from tplinker.metrics import F1, Precision, Recall, SampleAccuracy


class MetricsTest(unittest.TestCase):

    def test_precision(self):
        p = Precision()
        v = p.compute()
        print(v)


if __name__ == "__main__":
    unittest.main()
