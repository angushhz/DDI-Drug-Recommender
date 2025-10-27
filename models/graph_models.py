# models/graph_models.py
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Construct a layernorm module in the TF style (epsilon inside the square root).
    """
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FuseEmbeddings(nn.Module):
    """Construct the embeddings from word, visit and token_type embeddings.
    This is a simplified version that behaves like BertEmbeddings for compatibility.
    """

    def __init__(self, config, dx_voc=None, rx_voc=None):
        super(FuseEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # LayerNorm is not snake-cased to stick with TensorFlow model variable name
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Store vocabularies for potential future use
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        embeddings = words_embeddings + \
            self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings