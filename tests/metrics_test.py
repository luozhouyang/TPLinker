import unittest

from tplinker.metrics import F1, Precision, Recall, SampleAccuracy


class MetricsTest(unittest.TestCase):

    def test_precision(self):
        p = Precision('data/tplinker/bert/rel2id.json', max_sequence_length=100)
        v = p.compute()
        print(v)


if __name__ == "__main__":
    unittest.main()
