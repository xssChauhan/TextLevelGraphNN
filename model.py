import torch.nn as nn
import torch
# Add reference

class GNNModel(nn.Module):
    def __init__(self, num_node, embedding_dim, num_cls, pre_embed=None):
        super(GNNModel, self).__init__()
        if pre_embed is not None:
            self.node_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pre_embed), freeze=False)
        else:
            self.node_embedding = nn.Embedding(num_node, embedding_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.node_embedding.weight)

        # self.edge_weight = nn.Embedding(num_node * num_node+1, 1, padding_idx=0)       # vocab excludes <pad>
        self.edge_weight = nn.Embedding((num_node-1)*(num_node-1)+1, 1, padding_idx=0)   # vocab includes <pad>
        self.node_weight = nn.Embedding(num_node, 1, padding_idx=0)  # gate control
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, num_cls),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LogSoftmax(dim=1)
        )

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.edge_weight.weight)
        nn.init.xavier_uniform_(self.node_weight.weight)

    def forward(self, X, NX, EW):
        '''
        :param X: (bz, max_seq_len)  sentence nodes
        :param NX: (bz, max_seq_len, neighbor_num)  neighbor nodes of each node in X
        :param EW: (bz, max_seq_len, neighbor_num)  neighbor weights of each node in X
        :return:
        '''
        # neighbor (bz, seq_len, neighbor_num, embed_dim)
        Ra = self.node_embedding(NX)
        # edge weight  (bz, seq_len, neighbor_num, 1)
        Ean = self.edge_weight(EW)
        # neighbor representation  (bz, seq_len, embed_dim)
        Mn = (Ra * Ean).max(dim=2)[0]    # max pooling
        # self representation (bz, seq_len, embed_dim)
        Rn = self.node_embedding(X)
        # self node weight  (bz, seq_len, 1)
        Nn = self.node_weight(X)
        # aggregate node features
        y = (1 - Nn) * Mn + Nn * Rn
        y = self.fc(y.sum(dim=1))
        return y
