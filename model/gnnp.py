'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.autograd import Variable
import numpy as np


class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, gu_dict, norm_rgi, norm_rgu, norm_rui, embedding_dim, layer_size, drop_ratio):
        super(AGREE, self).__init__()
        self.useCuda = True
        self.n_users = num_users
        self.n_groups = num_groups
        self.n_items = num_items

        self.gu_dict = gu_dict

        self.norm_rgi = self.create_sparse_tensor_for_coo(norm_rgi)
        self.norm_rgu = self.create_sparse_tensor_for_coo(norm_rgu)
        self.norm_rui = self.create_sparse_tensor_for_coo(norm_rui)

        self.drop_ratio = drop_ratio

        self.layer_size = layer_size
        self.n_layer = len(layer_size)

        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        #self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.GNNLayers = nn.ModuleList().cuda()

        for From, To in zip(self.layer_size[:-1], self.layer_size[1:]):
            self.GNNLayers.append(GNNLayer(From, To, self.drop_ratio))

        #self.predictlayer = PredictLayer(3 * embedding_dim, self.drop_ratio)

        #print(self.modules())
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal(m.weight)

    def forward(self, group_inputs, item_inputs):
        gidx = Variable(torch.LongTensor([i for i in range(self.n_groups)]))
        uidx = Variable(torch.LongTensor([i for i in range(self.n_users)]))
        iidx = Variable(torch.LongTensor([i for i in range(self.n_items)]))
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()
            gidx = gidx.cuda()

        #group_embedding = self.groupembeds(gidx).data
        #user_embedding = self.userembeds(uidx).data
        #item_embedding = self.itemembeds(iidx).data
        group_embedding = self.groupembeds(gidx)
        user_embedding = self.userembeds(uidx)
        item_embedding = self.itemembeds(iidx)
        #print(self.groupembeds(gidx).data)
        #print(self.groupembeds(gidx))

        final_group_embed = group_embedding.clone()
        final_item_embed = item_embedding.clone()
        for gnn in self.GNNLayers:
            group_embedding, user_embedding, item_embedding = gnn(group_embedding, user_embedding, item_embedding, self.gu_dict, self.norm_rgi, self.norm_rgu, self.norm_rui)
            #group_embedding = nn.LeakyReLU()(group_embedding)
            #item_embedding = nn.LeakyReLU()(item_embedding)
            #user_embedding = nn.LeakyReLU()(user_embedding)
            final_group_embed = torch.cat([final_group_embed, group_embedding.clone()],dim=1)
            final_item_embed = torch.cat([final_item_embed, item_embedding.clone()], dim=1)

        #print(len(group_inputs))
        group_inputs_batch = Variable(torch.LongTensor(group_inputs))
        item_inputs_batch = Variable(torch.LongTensor(item_inputs))
        #print(final_group_embed.size())
        #print(final_group_embed[Variable(torch.LongTensor([1,1,1,1]))])
        #print(final_group_embed)
        group_embeds = final_group_embed[group_inputs_batch]
        #print(group_embeds)
        item_embeds_full = final_item_embed[item_inputs_batch]
        #print("group_embeds: "+str(group_embeds.size())) #1024*192
        #print("item_embeds: "+str(item_embeds_full.size())) #1024*192
        """
        group_embeds = Variable(torch.Tensor())
        item_embeds_full = self.itemembeds(Variable(torch.LongTensor(item_inputs)))
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i]
            members_embeds = self.userembeds(Variable(torch.LongTensor(members)))
            items_numb = []
            for _ in members:
                items_numb.append(j)
            item_embeds = self.itemembeds(Variable(torch.LongTensor(items_numb)))
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
            at_wt = self.attention(group_item_embeds)
            g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            group_embeds_pure = self.groupembeds(Variable(torch.LongTensor([i])))
            g_embeds = g_embeds_with_attention + group_embeds_pure
            if group_embeds.dim() == 0:
                group_embeds = g_embeds
            else:
                group_embeds = torch.cat((group_embeds, g_embeds))
        """

        # For sigmoid loss
        """
        element_embeds = torch.mul(group_embeds, item_embeds_full)  # Element-wise product
        #new_embeds = element_embeds
        #new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full), dim=1)
        sig = nn.Sigmoid()
        y = sig(self.predictlayer(element_embeds)) """

        # For BPR loss
        #score = torch.matmul(group_embeds, item_embeds_full.transpose(0, 1))
        score = torch.mul(group_embeds, item_embeds_full)
        #print("group_embeds: "+str(group_embeds))
        #print("item_embeds: "+str(item_embeds_full))
        #print("scores: "+str(score))
        y = torch.sum(score, dim=1)
        #print("y:"+str(y))
        #print("y size:"+str(y.size()))#1024
        return y

    # user forward
    """
    def usr_forward(self, user_inputs, item_inputs):
        user_inputs_var, item_inputs_var = Variable(user_inputs), Variable(item_inputs)
        #print(user_inputs_var)
        #print(item_inputs_var)
        user_embeds = self.userembeds(user_inputs_var)
        #print(user_embeds)
        item_embeds = self.itemembeds(item_inputs_var)
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = F.sigmoid(self.predictlayer(new_embeds))
        return y
    """
    def create_sparse_tensor_for_coo(self,X):
        values = X.data
        indices = np.vstack((X.row, X.col))
        indices = indices.astype(np.int64)
        values = values.astype(np.float32)
        i = torch.from_numpy(indices)
        v = torch.from_numpy(values)
        shape = X.shape
        return torch.sparse.FloatTensor(i,v,torch.Size(shape))


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim).cuda()

    def forward(self, user_inputs):
        #print("forward")
        #print(user_inputs)
        user_inputs=user_inputs.cuda()
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim).cuda()

    def forward(self, item_inputs):
        item_inputs=item_inputs.cuda()
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim).cuda()

    def forward(self, num_group):
        num_group=num_group.cuda()
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim/2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(embedding_dim/2, 1),
        ).cuda()

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out, dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        ).cuda()

    def forward(self, x):
        out = self.linear(x)
        return out

class GNNLayer(nn.Module):
    def __init__(self, inF, outF, drop_ratio):
        super(GNNLayer, self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear_i1 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_i2 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_i3 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_i4 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_i5 = nn.Linear(in_features=inF, out_features=outF).cuda()

        self.linear_g1 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_g2 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_g3 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_g4 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_g5 = nn.Linear(in_features=inF, out_features=outF).cuda()

        self.linear_u1 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_u2 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_u3 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_u4 = nn.Linear(in_features=inF, out_features=outF).cuda()
        self.linear_u5 = nn.Linear(in_features=inF, out_features=outF).cuda()

        self.re_g = nn.LeakyReLU()
        self.re_u = nn.LeakyReLU()
        self.re_i = nn.LeakyReLU()

        self.drop_g = nn.Dropout(drop_ratio)
        self.drop_u = nn.Dropout(drop_ratio)
        self.drop_i = nn.Dropout(drop_ratio)

    def forward(self, group_embedding, user_embedding, item_embedding, gu_dict, norm_rgi, norm_rgu, norm_rui):
        rgi = norm_rgi
        rgu = norm_rgu
        rui = norm_rui
        rgi = rgi.cuda()
        rgu = rgu.cuda()
        rui = rui.cuda()
        group_embedding = group_embedding.cuda()
        user_embedding = user_embedding.cuda()
        item_embedding = item_embedding.cuda()

        embed_rui_ei = torch.sparse.mm(rui, item_embedding).cuda()
        embed_rgu_t_eg = torch.sparse.mm(rgu.transpose(0, 1), group_embedding)
        new_user_embedding = self.linear_u1(user_embedding) + self.linear_u2(embed_rui_ei) + self.linear_u3(embed_rui_ei * user_embedding) + \
                             self.linear_u4(embed_rgu_t_eg * user_embedding) + self.linear_u5(embed_rgu_t_eg)

        #print(item_embedding.size()) #1490*8


        all_attention = Variable(torch.Tensor())
        for i in gu_dict:
            members = gu_dict[i]
            members_tensor = Variable(torch.LongTensor(members)).cuda()
            members_embeds = user_embedding[members_tensor]
            #print(members_embeds.size()) #1184,8
            e_u = torch.matmul(item_embedding, torch.t(members_embeds))
            #print(e_u.size()) #1490 * 1184
            weight = F.softmax(e_u, dim=0)
            attention = torch.matmul(weight, members_embeds) #1490*8
            if len(all_attention) == 0:
                all_attention = attention
            else:
                all_attention = all_attention + attention #1490*8

        attentive_item = all_attention * item_embedding

        embed_rgi_t_eg = torch.sparse.mm(rgi.transpose(0,1), group_embedding)
        embed_rui_t_eu = torch.sparse.mm(rui.transpose(0, 1), user_embedding)
        new_item_embedding = self.linear_i1(item_embedding)+self.linear_i2(embed_rui_t_eu)+self.linear_i3(embed_rui_t_eu*item_embedding)+\
                             self.linear_i4(embed_rgi_t_eg*item_embedding)+self.linear_i5(embed_rgi_t_eg)

        embed_rgi_ei = torch.sparse.mm(rgi, item_embedding).cuda()
        embed_rgu_eu = torch.sparse.mm(rgu, user_embedding).cuda()
        embed_atten_item = torch.sparse.mm(rgi, attentive_item).cuda()

        new_group_embedding = self.linear_g1(group_embedding) + self.linear_g2(embed_rgi_ei) + self.linear_g3(embed_rgu_eu * group_embedding) +\
                              self.linear_g4(embed_rgi_ei * group_embedding) + self.linear_g5(embed_atten_item)

        #new_group_embedding = self.linear_g1(group_embedding) + self.linear_g2(embed_rgi_ei) + \
        #                      self.linear_g3(embed_rgu_eu * group_embedding) + self.linear_g4(embed_rgi_ei * group_embedding)


        new_group_embedding = self.re_g(new_group_embedding)
        new_user_embedding = self.re_u(new_user_embedding)
        new_item_embedding = self.re_i(new_item_embedding)

        new_group_embedding = self.drop_g(new_group_embedding)
        new_user_embedding = self.drop_u(new_user_embedding)
        new_item_embedding = self.drop_i(new_item_embedding)

        group_norm = nn.functional.normalize(new_group_embedding, dim=1, p=2)
        user_norm = nn.functional.normalize(new_user_embedding, dim=1, p=2)
        item_norm = nn.functional.normalize(new_item_embedding, dim=1, p=2)

        return group_norm, user_norm, item_norm
