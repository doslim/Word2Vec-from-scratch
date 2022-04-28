import torch.nn as nn
import torch


class Finetune_Model(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embed_size_1: int,
                 embed_size_2: int,
                 dropout: float,
                 max_norm: float,
                 embedding):
        super(Finetune_Model, self).__init__()
        self.embeddings_1 = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size_1,
            max_norm=1
        )
        self.embeddings_1.weight = nn.Parameter(embedding)
        self.embeddings_1.weight.requires_grad = False

        self.embeddings_2 = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size_2,
            max_norm=max_norm
        )
        nn.init.constant_(self.embeddings_2.weight, 0)

        #         self.linear_1 = nn.Linear(
        #             in_features=embed_size_1 + embed_size_2,
        #             out_features=128,
        #         )
        #         self.linear_2 = nn.Linear(
        #             in_features=embed_size_1 + embed_size_2,
        #             out_features=128
        #         )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * (embed_size_1 + embed_size_2), 1),
                                        nn.Sigmoid())

    def forward(self, word1, word2):
        x1 = torch.cat((self.embeddings_1(word1), self.embeddings_2(word1)), dim=1)
        x2 = torch.cat((self.embeddings_1(word2), self.embeddings_2(word2)), dim=1)

        # x1 = self.linear_1(x1)
        # x2 = self.linear_2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x.squeeze()
