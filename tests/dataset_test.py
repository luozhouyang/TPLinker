import unittest

import torch
from tplinker.dataset import TPLinkerBertDataset


class DatasetTest(unittest.TestCase):

    def test_bert_dataset(self):
        ds = TPLinkerBertDataset(
            input_files=['data/tplinker/bert/valid_data.jsonl'],
            pretrained_bert_path='data/bert-base-cased',
            rel2id_path='data/tplinker/bert/rel2id.json',
            max_sequence_length=100)

        dl = torch.utils.data.DataLoader(ds, batch_size=2, drop_last=True)
        for idx, d in enumerate(dl):
            if idx == 10:
                break
            print()
            print(d)


if __name__ == "__main__":
    unittest.main()
