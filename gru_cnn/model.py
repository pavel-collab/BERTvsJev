"""Frozen fastText vectors → independent forward/backward GRU → parallel CNN → heads.

Adapted from page-stream-segmentation/app/pss/model.py; see LICENSE.source.
"""
import re
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from gru_cnn.common import LABELS

CONFIG = dict(embedding_dim=300, hidden_dim=128, channels=200,
              representation_dim=128, dropout=0.5, max_length=150, embedding_source="fasttext")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class FastTextEncoder:
    """Fixed pretrained vectors, including subword vectors for unseen tokens."""
    def __init__(self, path, dimension, expected_sha256=None):
        import fasttext
        self.sha256 = file_sha256(path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('SHA-256 fastText не совпадает с checkpoint')
        self.ft = fasttext.load_model(str(path))
        self.dimension = self.ft.get_dimension()
        if self.dimension != dimension:
            raise ValueError('Размерность fastText не совпадает с конфигурацией')
        self.cache = {}

    def encode(self, texts, max_length):
        sequences = [tokenize(text, max_length) for text in texts]
        lengths = torch.tensor([max(1, len(tokens)) for tokens in sequences])
        x = torch.zeros(len(texts), int(lengths.max()), self.dimension)
        for i, tokens in enumerate(sequences):
            for j, token in enumerate(tokens):
                if token not in self.cache:
                    self.cache[token] = torch.tensor(self.ft.get_word_vector(token))
                x[i, j] = self.cache[token]
        return x, lengths


def encode_inputs(model, vocab, texts):
    return model.fasttext_encoder.encode(texts, model.config['max_length'])


def tokenize(text, max_length=150):
    text = re.sub(r'http\S+', 'URL', text)
    tokens = [part.strip().lower() for part in re.findall(r'[#\w-]+|[^#\w-]+', text.strip())
              if part.strip()]
    if len(tokens) > max_length:
        first = max_length // 2
        tokens = tokens[:first] + tokens[-(max_length - first):]
    return tokens


def build_vocab(rows, max_length=150):
    tokens = sorted({token for row in rows for token in tokenize(row['text'], max_length)})
    return {'<pad>': 0, '<unk>': 1, **{token: i + 2 for i, token in enumerate(tokens)}}



class GRUCNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(CONFIG if config is None else config)
        c = self.config
        self.gru_forward = nn.GRU(c['embedding_dim'], c['hidden_dim'], batch_first=True)
        self.gru_backward = nn.GRU(c['embedding_dim'], c['hidden_dim'], batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(2 * c['hidden_dim'], c['channels'], size) for size in (3, 4, 5)
        ])
        self.dropout = nn.Dropout(c['dropout'])
        self.dense = nn.Linear(3 * c['channels'], c['representation_dim'])
        self.heads = nn.ModuleDict({task: nn.Linear(c['representation_dim'], len(labels))
                                    for task, labels in LABELS.items()})

    def input_dropout(self, x):
        mask = F.dropout(torch.ones_like(x[:, :1, :]), self.config['dropout'], self.training)
        return x * mask

    def forward(self, ids, lengths):
        x = ids
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        valid = positions < lengths.to(ids.device)[:, None]
        reverse = (lengths.to(ids.device)[:, None] - 1 - positions).clamp_min(0)
        reverse = reverse[:, :, None].expand_as(x)
        forward, _ = self.gru_forward(pack_padded_sequence(
            self.input_dropout(x), lengths.cpu(), batch_first=True, enforce_sorted=False))
        backward, _ = self.gru_backward(pack_padded_sequence(
            self.input_dropout(x).gather(1, reverse), lengths.cpu(),
            batch_first=True, enforce_sorted=False))
        forward, _ = pad_packed_sequence(forward, batch_first=True, total_length=ids.shape[1])
        backward, _ = pad_packed_sequence(backward, batch_first=True, total_length=ids.shape[1])
        backward = backward.gather(1, reverse[:, :, :1].expand(-1, -1, backward.shape[2]))
        features = torch.cat((forward, backward), dim=2).masked_fill(~valid[:, :, None], 0)
        pooled = []
        for conv in self.convs:
            # Explicit SAME padding, including the asymmetric width-4 case.
            padding = conv.kernel_size[0] - 1
            y = conv(F.pad(features.transpose(1, 2), (padding // 2, padding - padding // 2)))
            y = F.leaky_relu(y, negative_slope=0.3).masked_fill(~valid[:, None, :], -torch.inf)
            pooled.append(self.dropout(y.amax(dim=2)))
        representation = F.leaky_relu(self.dense(torch.cat(pooled, dim=1)), negative_slope=0.3)
        return {task: head(representation) for task, head in self.heads.items()}


@torch.inference_mode()
def predict(model, vocab, texts):
    model.eval()
    ids, lengths = encode_inputs(model, vocab, texts)
    logits = model(ids, lengths)
    records = [{'predictions': {}, 'probabilities': {}} for _ in texts]
    for task, labels in LABELS.items():
        for record, probs in zip(records, logits[task].softmax(-1).tolist()):
            record['predictions'][task] = labels[max(range(len(probs)), key=probs.__getitem__)]
            record['probabilities'][task] = dict(zip(map(str, labels), probs))
    return records


def load_model(path, fasttext_path=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['labels'] != LABELS:
        raise ValueError('Схема меток модели отличается от текущей')
    if checkpoint['config'].get('embedding_source') != 'fasttext' or not checkpoint.get('fasttext'):
        raise ValueError('Checkpoint должен использовать предобученный fastText')
    model = GRUCNN(checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.fasttext_encoder = FastTextEncoder(
        fasttext_path or checkpoint['fasttext']['path'], model.config['embedding_dim'],
        checkpoint['fasttext']['sha256'])
    model.eval()
    return model, checkpoint['vocab']
