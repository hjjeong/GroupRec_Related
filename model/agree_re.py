import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()

        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, item_inputs):
        group_embed = self.groupembeds(group_inputs)
        item_embed = self.itemembeds(item_inputs)
        ncf_input = torch.cat((group_embed*item_embed, group_embed, item_embed), dim=-1)
        y = torch.sigmoid(self.predictlayer(ncf_input)) #BPR
        return y


class GR_NCF(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.group_member_dict = group_member_dict
        
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        #self.user_encoder = nn.Linear(embedding_dim, 1, bias=False)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 2)
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):

        user_aggregation = []
        for i in group_inputs.tolist():
            members = self.group_member_dict[i]
            uidx = Variable(torch.LongTensor(members)).cuda()
            members_embeds = self.userembeds(uidx)

            user_aggregation.append(torch.relu(torch.mean(members_embeds, dim=0)))
            '''
            members_weights = self.user_encoder(members_embeds)
            members_weights = torch.softmax(members_weights, 0)
            members_embeds = members_embeds.transpose(0,1)@members_weights
            user_aggregation.append(members_embeds.squeeze(-1))
            '''

        user_aggregation = torch.stack(user_aggregation)

        group_embed = self.group_encoder(user_aggregation)

        z_mu, log_sigma = torch.split(group_embed, self.z_dim, dim=-1)
        z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        
        
        item_embed = self.itemembeds(item_inputs)
        

        out = None
        if is_training:
            group_embed_train = self.groupembeds(group_inputs)
            z = group_embed_train
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) 

            dkl = torch.mean(torch.sum((group_embed_train-z_mu)**2, dim=-1, keepdim=True))
            out = [y, dkl]
        else:
            self.temp_z_mu = z_mu.detach()
            self.temp_z_sigma = z_sigma.detach()
            
            z = z_mu #self.sample_z(z_mu, z_sigma)
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out

class VARGR_NCF2(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.group_member_dict = group_member_dict
        
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        #self.user_encoder = nn.Linear(embedding_dim, 1, bias=False)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 1)
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)
        # initial model
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):

        user_aggregation = []
        for i in group_inputs.tolist():
            members = self.group_member_dict[i]
            uidx = Variable(torch.LongTensor(members)).cuda()
            members_embeds = self.userembeds(uidx)

            user_aggregation.append(torch.relu(torch.mean(members_embeds, dim=0)))
            '''
            members_weights = self.user_encoder(members_embeds)
            members_weights = torch.softmax(members_weights, 0)
            members_embeds = members_embeds.transpose(0,1)@members_weights
            user_aggregation.append(members_embeds.squeeze(-1))
            '''

        user_aggregation = torch.stack(user_aggregation)

        group_embed = self.group_encoder(user_aggregation)

        z_mu, log_sigma = torch.split(group_embed, self.z_dim, dim=-1)
        z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        
        
        item_embed = self.itemembeds(item_inputs)
        

        out = None
        if is_training:
            group_embed_train = self.groupembeds(group_inputs)
            std = torch.randn(group_embed_train.detach().shape).cuda()
            z = group_embed_train+ self.q_std*std
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) 
            
            z = z_mu+z_sigma*std
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y2 = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            
            dkl = torch.mean(torch.sum(self._KL(group_embed_train, self.q_std*torch.ones(z_sigma.detach().shape).cuda(), z_mu, z_sigma), dim=-1, keepdim=True))
            out = [y, y2, dkl]
        else:
            self.temp_z_mu = z_mu.detach()
            self.temp_z_sigma = z_sigma.detach()
            
            z = self.sample_z(z_mu, z_sigma)
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out

class VARGR_NCF(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.group_member_dict = group_member_dict
        
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        
        self.predictlayer = PredictLayer(3 * embedding_dim)
        #self.user_encoder = nn.Linear(embedding_dim, 1, bias=False)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 1)
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):

        user_aggregation = []
        for i in group_inputs.tolist():
            members = self.group_member_dict[i]
            uidx = Variable(torch.LongTensor(members)).cuda()
            members_embeds = self.userembeds(uidx)

            user_aggregation.append(torch.relu(torch.mean(members_embeds, dim=0)))
            '''
            members_weights = self.user_encoder(members_embeds)
            members_weights = torch.softmax(members_weights, 0)
            members_embeds = members_embeds.transpose(0,1)@members_weights
            user_aggregation.append(members_embeds.squeeze(-1))
            '''

        user_aggregation = torch.stack(user_aggregation)

        group_embed = self.group_encoder(user_aggregation)

        z_mu, log_sigma = torch.split(group_embed, self.z_dim, dim=-1)
        z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        
        
        item_embed = self.itemembeds(item_inputs)
        

        out = None
        if is_training:
            group_embed_train = self.groupembeds(group_inputs)
            z = group_embed_train+ self.q_std*torch.randn(group_embed_train.detach().shape).cuda()
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) 

            dkl = torch.mean(torch.sum(self._KL(group_embed_train, self.q_std*torch.ones(z_sigma.detach().shape).cuda(), z_mu, z_sigma), dim=-1, keepdim=True))
            out = [y, dkl]
        else:
            self.temp_z_mu = z_mu.detach()
            self.temp_z_sigma = z_sigma.detach()
            
            z = self.sample_z(z_mu, z_sigma)
            ncf_input = torch.cat((z*item_embed, z, item_embed), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out



class GR_AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)

        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        self.user_encoder = MLP(embedding_dim, embedding_dim, 1)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 1)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)

        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):
         
        item_embeds_full = self.itemembeds(item_inputs)
        
        
        out = None
        if is_training:
            user_aggregation = []
            group_embeds = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs.tolist()):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                members_encode = self.user_encoder(members_embeds)
                user_aggregation.append(torch.relu(torch.mean(members_encode, dim=0)))
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = self.groupembeds(g_em)
                group_embeds_pure = group_embeds_pure #+ self.q_std*torch.randn(group_embeds_pure.detach().shape).cuda()
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))

            user_aggregation = torch.stack(user_aggregation)
            group_z = self.group_encoder(user_aggregation)
            z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
            #z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)       

            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR

            dkl = torch.mean(torch.sum((self.groupembeds(group_inputs)-z_mu)**2, dim=-1, keepdim=True))
            out = [y, dkl]
        else:
            z_mus = []
            z_sigmas = []
            group_embeds = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                members_encode = self.user_encoder(members_embeds)
                group_z = torch.relu(torch.mean(members_encode, dim=0))
                group_z = self.group_encoder(group_z)
                z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
                z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
                z_mus.append(z_mu)
                z_sigmas.append(z_sigma)
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                #g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = z_mu #self.sample_z(z_mu, z_sigma)
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))

            z_mus = torch.stack(z_mus)
            z_sigmas = torch.stack(z_sigmas)

            self.temp_z_mu = z_mus.detach()
            self.temp_z_sigma = z_sigmas.detach()
            
            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out

class VARGR_AGREE2(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)

        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        self.user_encoder = MLP(embedding_dim, embedding_dim, 1)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 1)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)

        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):
         
        item_embeds_full = self.itemembeds(item_inputs)
        
        
        out = None
        if is_training:
            z_mus = []
            z_sigmas = []
            group_embeds = Variable(torch.Tensor()).cuda()
            group_embeds2 = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs.tolist()):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                members_encode = self.user_encoder(members_embeds)
                group_z = torch.relu(torch.mean(members_encode, dim=0))
                group_z = self.group_encoder(group_z)
                z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
                z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
                z_mus.append(z_mu)
                z_sigmas.append(z_sigma)
                #user_aggregation.append(torch.relu(torch.mean(members_encode, dim=0)))
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = self.groupembeds(g_em)
                std = torch.randn(group_embeds_pure.detach().shape).cuda()
                group_embeds_pure = group_embeds_pure+ self.q_std*std
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))
                group_embeds_pure2 = z_mu+ z_sigma*std
                g_embeds2 = g_embeds_with_attention + group_embeds_pure2
                if group_embeds.dim() == 0:
                    group_embeds2 = g_embeds2
                else:
                    group_embeds2 = torch.cat((group_embeds2, g_embeds2))
            z_mus = torch.stack(z_mus)
            z_sigmas = torch.stack(z_sigmas)     

            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            
            ncf_input = torch.cat((group_embeds2*item_embeds_full, group_embeds2, item_embeds_full), dim=-1)
            y2 = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            
            dkl = torch.mean(torch.sum(self._KL(self.groupembeds(group_inputs), self.q_std*torch.ones(z_sigma.detach().shape).cuda(), z_mus, z_sigmas), dim=-1, keepdim=True))
            out = [y, y2, dkl]
        else:
            z_mus = []
            z_sigmas = []
            group_embeds = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                members_encode = self.user_encoder(members_embeds)
                group_z = torch.relu(torch.mean(members_encode, dim=0))
                group_z = self.group_encoder(group_z)
                z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
                z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
                z_mus.append(z_mu)
                z_sigmas.append(z_sigma)
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                #g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = self.sample_z(z_mu, z_sigma)
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))

            z_mus = torch.stack(z_mus)
            z_sigmas = torch.stack(z_sigmas)

            self.temp_z_mu = z_mus.detach()
            self.temp_z_sigma = z_sigmas.detach()
            
            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out


class VARGR_AGREE3(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)

        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        #self.user_encoder = MLP(embedding_dim, embedding_dim, 1)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 1)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)

        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape).cuda()

        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):
         
        item_embeds_full = self.itemembeds(item_inputs)
        
        
        out = None
        if is_training:
            z_mus = []
            z_sigmas = []
            group_embeds = Variable(torch.Tensor()).cuda()
            group_embeds2 = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs.tolist()):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                #members_encode = self.user_encoder(members_embeds)
                #group_z = torch.relu(torch.mean(members_encode, dim=0))
                group_z = torch.relu(torch.mean(members_embeds, dim=0))
                group_z = self.group_encoder(group_z)
                z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
                z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
                z_mus.append(z_mu)
                z_sigmas.append(z_sigma)
                #user_aggregation.append(torch.relu(torch.mean(members_encode, dim=0)))
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                g_em = Variable(torch.LongTensor([i]))
                group_embeds_pure = self.groupembeds(g_em)
                std = torch.randn(group_embeds_pure.detach().shape).cuda()
                group_embeds_pure = group_embeds_pure+ self.q_std*std
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))
                group_embeds_pure2 = z_mu+ z_sigma*std
                g_embeds2 = g_embeds_with_attention + group_embeds_pure2
                if group_embeds.dim() == 0:
                    group_embeds2 = g_embeds2
                else:
                    group_embeds2 = torch.cat((group_embeds2, g_embeds2))
            z_mus = torch.stack(z_mus)
            z_sigmas = torch.stack(z_sigmas)     

            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            
            ncf_input = torch.cat((group_embeds2*item_embeds_full, group_embeds2, item_embeds_full), dim=-1)
            y2 = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            
            dkl = torch.mean(torch.sum(self._KL(self.groupembeds(group_inputs), self.q_std*torch.ones(z_sigma.detach().shape).cuda(), z_mus, z_sigmas), dim=-1, keepdim=True))
            out = [y, y2, dkl]
        else:
            z_mus = []
            z_sigmas = []
            group_embeds = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                #members_encode = self.user_encoder(members_embeds)
                #group_z = torch.relu(torch.mean(members_encode, dim=0))
                group_z = torch.relu(torch.mean(members_embeds, dim=0))
                group_z = self.group_encoder(group_z)
                z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
                z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
                z_mus.append(z_mu)
                z_sigmas.append(z_sigma)
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                #g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = self.sample_z(z_mu, z_sigma)
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))

            z_mus = torch.stack(z_mus)
            z_sigmas = torch.stack(z_sigmas)

            self.temp_z_mu = z_mus.detach()
            self.temp_z_sigma = z_sigmas.detach()
            
            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out

class VARGR_AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super().__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)

        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        self.user_encoder = MLP(embedding_dim, embedding_dim, 1)
        self.group_encoder = MLP(embedding_dim, embedding_dim*2, 1)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
        
        
        self.z_dim = embedding_dim
        self.q_std = np.sqrt(2/embedding_dim)

        self.temp_z_mu = None
        self.temp_z_sigma = None
    
    def sample_z(self, mu, sigma):
        """Reparametrization trick
        """
        eps = torch.randn(sigma.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + sigma * eps
    
    def get_z_dist(self):
        return self.temp_z_mu, self.temp_z_sigma
    
    def _KL(self, mu, sigma, mup, sigmap):
        t = 2*torch.log(sigmap) - 2*torch.log(sigma) + (sigma**2)* (1/sigmap**2) + (1/sigmap**2)*((mu-mup)**2)-1
        return t*0.5
    
    def forward(self, group_inputs, item_inputs, is_training=False):
         
        item_embeds_full = self.itemembeds(item_inputs)
        
        
        out = None
        if is_training:
            user_aggregation = []
            group_embeds = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs.tolist()):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                members_encode = self.user_encoder(members_embeds)
                user_aggregation.append(torch.relu(torch.mean(members_encode, dim=0)))
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = self.groupembeds(g_em)
                group_embeds_pure = group_embeds_pure+ self.q_std*torch.randn(group_embeds_pure.detach().shape).cuda()
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))

            user_aggregation = torch.stack(user_aggregation)
            group_z = self.group_encoder(user_aggregation)
            z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
            z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)       

            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR

            dkl = torch.mean(torch.sum(self._KL(self.groupembeds(group_inputs), self.q_std*torch.ones(z_sigma.detach().shape).cuda(), z_mu, z_sigma), dim=-1, keepdim=True))
            out = [y, dkl]
        else:
            z_mus = []
            z_sigmas = []
            group_embeds = Variable(torch.Tensor()).cuda()
            for i, j in zip(group_inputs.tolist(), item_inputs):
                members = self.group_member_dict[i]
                uidx = Variable(torch.LongTensor(members)).cuda()
                members_embeds = self.userembeds(uidx)
                members_encode = self.user_encoder(members_embeds)
                group_z = torch.relu(torch.mean(members_encode, dim=0))
                group_z = self.group_encoder(group_z)
                z_mu, log_sigma = torch.split(group_z, self.z_dim, dim=-1)
                z_sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
                z_mus.append(z_mu)
                z_sigmas.append(z_sigma)
                items_numb = []
                for _ in members:
                    items_numb.append(j)
                target_item = Variable(torch.LongTensor(items_numb)).cuda()
                item_embeds = self.itemembeds(target_item)
                group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
                at_wt = self.attention(group_item_embeds)
                g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
                #g_em = Variable(torch.LongTensor([i])).cuda()
                group_embeds_pure = self.sample_z(z_mu, z_sigma)
                g_embeds = g_embeds_with_attention + group_embeds_pure
                if group_embeds.dim() == 0:
                    group_embeds = g_embeds
                else:
                    group_embeds = torch.cat((group_embeds, g_embeds))

            z_mus = torch.stack(z_mus)
            z_sigmas = torch.stack(z_sigmas)

            self.temp_z_mu = z_mus.detach()
            self.temp_z_sigma = z_sigmas.detach()
            
            ncf_input = torch.cat((group_embeds*item_embeds_full, group_embeds, item_embeds_full), dim=-1)
            y = torch.sigmoid(self.predictlayer(ncf_input)) #noraml with BPR
            out = y
        return out


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim).cuda()

    def forward(self, user_inputs):
        #print("forward")
        #print(user_inputs)
        user_inputs = user_inputs.cuda()
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim).cuda()

    def forward(self, item_inputs):
        item_inputs = item_inputs.cuda()
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim).cuda()

    def forward(self, num_group):
        num_group = num_group.cuda()
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
        ).cuda()

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        ).cuda()

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=1, drop_ratio=0):
        super().__init__()
        layers = []
        out_dim = input_dim
        hidden_dim = int((input_dim+output_dim)/2)
        for i in range(num_hidden):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_ratio)]
        # Last layer without a ReLU
        layers += [nn.Linear(out_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
                   
    def forward(self, x):
        return self.mlp(x)