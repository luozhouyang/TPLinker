import logging
import math

import numpy as np
import torch
import torch.nn as nn
from glove import Glove


class GloveEmbedding(nn.Module):

    def __init__(self, pretrained_embedding_path, vocab, embedding_size, freeze=False, **kwargs):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embedding_size = embedding_size
        self.glove = Glove.load(pretrained_embedding_path)
        self.embedding = nn.Embedding.from_pretrained(self._build_embedding_matrix(), freeze=freeze)

    def _build_embedding_matrix(self):
        matrix = np.random.normal(-1, 1, size=(self.vocab_size, self.embedding_size))
        count = 0
        for idx, token in self.vocab.items():
            if token in self.glove.dictionary:
                matrix[idx] = self.glove.word_vectors[self.glove.dictionary[token]]
                count += 1
        logging.info(f'Load {count} tokens from pretrained embedding table.')
        matrix = torch.tensor(matrix)
        return matrix

    def forward(self, input_ids, **kwargs):
        return self.embedding(input_ids)


class DistanceEmbedding(nn.Module):

    def __init__(self, max_positions=512, embedding_size=768, **kwargs):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_size = embedding_size
        self.dist_embedding = self._init_embedding_table()
        self.register_parameter('distance_embedding', self.dist_embedding)

    def _init_embedding_table(self):
        matrix = np.zeros([self.max_positions, self.embedding_size])
        for d in range(self.max_positions):
            for i in range(self.embedding_size):
                if i % 2 == 0:
                    matrix[d][i] = math.sin(d / 10000**(i / self.embedding_size))
                else:
                    matrix[d][i] = math.cos(d / 10000**((i - 1) / self.embedding_size))
        embedding_table = nn.Parameter(data=torch.tensor(matrix, requires_grad=False), requires_grad=False)
        return embedding_table

    def forward(self, inputs, **kwargs):
        """Distance embedding.

        Args:
            inputs: Tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            embedding: Tensor, shape (batch_size, 1+2+...+seq_len, embedding_size)
        """
        batch_size, seq_len = inputs.size()[0], inputs.size()[1]
        segs = []
        for index in range(seq_len, 0, -1):
            segs.append(self.dist_embedding[:index, :])
        segs = torch.cat(segs, dim=0)
        embedding = segs[None, :, :].repeat(batch_size, 1, 1)
        return embedding


class TaggingProjector(nn.Module):

    def __init__(self, hidden_size, num_relations, name='proj', **kwargs):
        super().__init__()
        self.name = name
        self.fc_layers = [nn.Linear(hidden_size, 3) for _ in range(num_relations)]
        for index, fc in enumerate(self.fc_layers):
            self.register_parameter('{}_weights_{}'.format(self.name, index), fc.weight)
            self.register_parameter('{}_bias_{}'.format(self.name, index), fc.bias)

    def forward(self, hidden, **kwargs):
        """Project hiddens to tags for each relation.

        Args:
            hidden: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)

        Returns:
            outputs: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, num_tags=3)
        """
        outputs = []
        for fc in self.fc_layers:
            outputs.append(fc(hidden))
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.softmax(outputs, dim=-1)
        return outputs


class ConcatHandshaking(nn.Module):

    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, **kwargs):
        """Handshaking.

        Args:
            hidden: Tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            handshaking_hiddens: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)
        """
        seq_len = hidden.size()[1]
        handshaking_hiddens = []
        for i in range(seq_len):
            _h = hidden[:, i, :]
            repeat_hidden = _h[:, None, :].repeat(1, seq_len - i, 1)
            visibl_hidden = hidden[:, i:, :]
            shaking_hidden = torch.cat([repeat_hidden, visibl_hidden], dim=-1)
            shaking_hidden = self.fc(shaking_hidden)
            shaking_hidden = torch.tanh(shaking_hidden)
            handshaking_hiddens.append(shaking_hidden)
        handshaking_hiddens = torch.cat(handshaking_hiddens, dim=1)
        return handshaking_hiddens


class TPLinker(nn.Module):

    def __init__(self, hidden_size, num_relations, max_positions=512, add_distance_embedding=False, **kwargs):
        super().__init__()
        self.handshaking = ConcatHandshaking(hidden_size)
        self.h2t_proj = nn.Linear(hidden_size, 2)
        self.h2h_proj = TaggingProjector(hidden_size, num_relations, name='h2hproj')
        self.t2t_proj = TaggingProjector(hidden_size, num_relations, name='t2tproj')
        self.add_distance_embedding = add_distance_embedding
        if self.add_distance_embedding:
            self.distance_embedding = DistanceEmbedding(max_positions, embedding_size=hidden_size)

    def forward(self, hidden, **kwargs):
        """TPLinker model forward pass.

        Args:
            hidden: Tensor, output of BERT or BiLSTM, shape (batch_size, seq_len, hidden_size)

        Returns:
            h2t_hidden: Tensor, shape (batch_size, 1+2+...+seq_len, 2),
                logits for entity recognization
            h2h_hidden: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, 3),
                logits for relation recognization
            t2t_hidden: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, 3),
                logits for relation recognization
        """
        handshaking_hidden = self.handshaking(hidden)
        h2t_hidden, rel_hidden = handshaking_hidden, handshaking_hidden
        if self.add_distance_embedding:
            h2t_hidden += self.distance_embedding(hidden)
            rel_hidden += self.distance_embedding(hidden)
        h2t_hidden = self.h2t_proj(h2t_hidden)
        h2h_hidden = self.h2h_proj(rel_hidden)
        t2t_hidden = self.t2t_proj(rel_hidden)
        return h2t_hidden, h2h_hidden, t2t_hidden
