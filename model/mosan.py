'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

class MoSAN(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.quserembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)

        self.attention = MultiHeadedAttention(embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, item_inputs):
        group_inputs = group_inputs.tolist()
        item_inputs = item_inputs.tolist()
        group_embeds = Variable(torch.Tensor()).cuda()
        iidx = Variable(torch.LongTensor(item_inputs)).cuda()
        item_embeds_full = self.itemembeds(iidx)
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i]
            uidx = Variable(torch.LongTensor(members)).cuda()
            members_embeds = self.userembeds(uidx)
            members_queries = self.quserembeds(uidx)
            attn_output = self.attention(members_queries.unsqueeze(0),members_embeds.unsqueeze(0),members_embeds.unsqueeze(0) )

            attn_output = torch.sum(attn_output, dim=1)
            
            g_embeds = attn_output
            if group_embeds.dim() == 0:
                group_embeds = g_embeds
            else:
                group_embeds = torch.cat((group_embeds, g_embeds))

        element_embeds = torch.mul(group_embeds, item_embeds_full)  # Element-wise product

        y = torch.sum(element_embeds, dim=1)
        return y


class MultiHeadedAttention(nn.Module):
    # refered from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    def __init__(self, d_model, h=1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model, bias=True)) for _ in range(4)])
        self.dropout = nn.Dropout(0.5)
        self.wt = nn.Linear(d_model, 1, bias=True)
        self.attn = None
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.linears:
            nn.init.normal_(l.weight, std=self.d_k**-0.5)
        
    def _attention(self, query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        numq = query.size(1)
        num_bat = query.size(0)
        query = self.wt(query).squeeze(-1) #B Q
        key = self.wt(key).squeeze(-1) # B K
        key = key.unsqueeze(1).repeat([1,numq ,1]) # B Q K
        scores = query + key #B Q K
        mask = (torch.eye(numq).cuda()).unsqueeze(0).repeat([num_bat, 1, 1])*(-99999) # B Q Q
        scores = scores+ mask # B Q K

        #scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        p_attn = torch.softmax(scores, dim = 1)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value):

        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [self.dropout(l(x)) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self._attention(query, key, value)
        
        # 3) "Concat" using a view and apply a final linear. 
        return self.dropout(self.linears[-1](x))
    


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        #print("forward")
        #print(user_inputs)
        user_inputs = user_inputs
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_inputs = item_inputs
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        num_group = num_group
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

