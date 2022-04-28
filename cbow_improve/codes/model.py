# model.py
# Define the CBOW model

import torch.nn as nn
# from utils.constants import EMBED_MAX_NORM
EMBED_MAX_NORM = 1

class CBOW_Model(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=embed_size,
            out_features=vocab_size,
        )
        self.embed_size = embed_size

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x
