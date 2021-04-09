import unittest

from preprocessor.adapters import NYTBertAdapter


class DatasetAdapterTest(unittest.TestCase):

    def test_nyt_dataset_adapter(self):
        adapter = NYTBertAdapter('data/bert-base-cased')
        input_file = '/mnt/nas/zhouyang.lzy/public-datasets/NYT/raw_valid.json'
        output_file = 'data/preprocess/nyt_valid.jsonl'
        adapter.adapte(input_file, output_file, limit=2)


if __name__ == "__main__":
    unittest.main()
