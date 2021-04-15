import unittest

import torch
from tplinker.models_torch import TPLinkerBert, TPLinkerBiLSTM
from transformers import BertTokenizerFast


class ModelsTest(unittest.TestCase):

    def test_tplinker_bert(self):
        m = TPLinkerBert('data/bert-base-cased', 24, add_dist_embedding=True)
        t = BertTokenizerFast.from_pretrained('data/bert-base-cased', add_special_tokens=False, do_lower_case=False)
        codes = t.encode_plus('I love NLP!', return_offsets_mapping=True, add_special_tokens=False)
        print(codes)
        input_ids, attn_mask, segment_ids = codes['input_ids'], codes['attention_mask'], codes['token_type_ids']
        seq_len = len(input_ids)

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attn_mask = torch.tensor([attn_mask], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        flat_seq_len = seq_len * (seq_len + 1) // 2

        h2t, h2h, t2t = m(input_ids, attn_mask, segment_ids)
        self.assertEqual([1, flat_seq_len, 2], list(h2t.size()))
        self.assertEqual([1, 24, flat_seq_len, 3], list(h2h.size()))
        self.assertEqual([1, 24, flat_seq_len, 3], list(t2t.size()))

    def test_tplinker_bilstm(self):
        words = []
        with open('data/bert-base-cased/vocab.txt', mode='rt', encoding='utf-8') as fin:
            for line in fin:
                words.append(line.rstrip('\n'))
        vocab = {}
        for idx, token in enumerate(words):
            vocab[idx] = token
        m = TPLinkerBiLSTM(24, 768, 768,
                           pretrained_embedding_path='data/glove_300_nyt.emb',
                           vocab=vocab,
                           embedding_size=300,
                           add_dist_embedding=True).float()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        h2t, h2h, t2t = m(input_ids)
        self.assertEqual([1, 8 * 9 // 2, 2], list(h2t.size()))
        self.assertEqual([1, 24, 8 * 9 // 2, 3], list(h2h.size()))
        self.assertEqual([1, 24, 8 * 9 // 2, 3], list(t2t.size()))


if __name__ == "__main__":
    unittest.main()
