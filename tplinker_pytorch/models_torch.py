import logging

import torch
import torch.nn as nn
from tokenizers import BertWordPieceTokenizer
from transformers import BertModel

from .layers_torch import GloveEmbedding, TPLinker


class TPLinkerBert(nn.Module):

    def __init__(self, bert_model_path, num_relations, add_distance_embedding=False, **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.tplinker = TPLinker(
            hidden_size=self.bert.config.hidden_size,
            num_relations=num_relations,
            add_distance_embedding=add_distance_embedding,
            max_positions=512)

    def forward(self, input_ids, attn_mask, token_type_ids, **kwargs):
        sequence_output = self.bert(input_ids, attn_mask, token_type_ids)[0]
        h2t_outputs, h2h_outputs, t2t_outputs = self.tplinker(sequence_output)
        return h2t_outputs, h2h_outputs, t2t_outputs


class TPLinkerBiLSTM(nn.Module):

    def __init__(self,
                 num_relations,
                 encoder_hidden_size,
                 deocder_hidden_size,
                 embedding_dropout_rate=0.1,
                 lstm_dropout_rate=0.1,
                 add_dist_embedding=False,
                 max_positions=512,
                 **kwargs):
        super().__init__()
        self.embedding = self._build_embedding(**kwargs)
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)
        self.encoder = nn.LSTM(
            kwargs['embedding_size'],
            encoder_hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True)
        self.decoder = nn.LSTM(
            encoder_hidden_size,
            deocder_hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True)
        self.lstm_dropout = nn.Dropout(lstm_dropout_rate)
        self.tplinker = TPLinker(
            hidden_size=deocder_hidden_size,
            num_relations=num_relations,
            max_positions=max_positions,
            add_dist_embedding=add_dist_embedding,
            **kwargs)

    def _build_embedding(self, **kwargs):
        pretrained_embedding_path = kwargs.get('pretrained_embedding_path', None)
        vocab = kwargs.get('vocab', None)
        embedding_size = kwargs.get('embedding_size', None)
        assert embedding_size, "embedding_size must be provided."
        if pretrained_embedding_path and vocab and embedding_size:
            logging.info('Load pretrained embedding...')
            embedding = GloveEmbedding(pretrained_embedding_path, vocab=vocab, embedding_size=embedding_size)
            return embedding
        vocab_size = kwargs.get('vocab_size', len(vocab) if vocab else None)
        if vocab_size and embedding_size:
            logging.info('Build embedding matrix...')
            embedding = nn.Embedding(vocab_size, embedding_size)
            return embedding
        raise ValueError('Not enough params to build emebdding layer.')

    def forward(self, input_ids, **kwargs):
        embedding = self.embedding(input_ids)
        embedding = self.embedding_dropout(embedding)
        encoder_outputs, _ = self.encoder(embedding)
        encoder_outputs = self.lstm_dropout(encoder_outputs)
        decoder_outputs, _ = self.decoder(encoder_outputs)
        decoder_outputs = self.lstm_dropout(decoder_outputs)
        h2t_outputs, h2h_outputs, t2t_outputs = self.tplinker(decoder_outputs)
        return h2t_outputs, h2h_outputs, t2t_outputs
