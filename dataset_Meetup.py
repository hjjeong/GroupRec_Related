'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from itertools import repeat
import datetime
from time import time

class GDataset(object):

    def __init__(self, user_path, group_path, user_in_group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives

        # user data

        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + ".train.rating") #ui
        #self.user_testRatings = self.load_rating_file_as_list(user_path + ".test.rating")
        #self.user_testNegatives = self.load_negative_file(user_path + ".test.negative")

        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data

        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + ".train.rating") #gi
        self.group_trainRatings = self.load_rating_file_as_dict(group_path + ".train.rating")
        #self.group_testRatings = self.load_rating_file_as_list(group_path + ".test.rating")
        self.group_testRatings = self.load_rating_file_as_dict(group_path + ".test.rating")
        #self.group_testNegatives = self.load_negative_file(group_path + ".test.negative")

        self.num_groups, self.num_items = self.group_trainMatrix.shape

        self.gu_dict = self.gen_group_member_dict(user_in_group_path) #gu_dict
        
        self.num_groups = len(self.gu_dict)


    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                line = line.replace(" ", "")
                a = line.split('\t')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    if len(m)==0:
                        continue
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def convert_gudict_to_matrix(self, gu):
        gu_mat = sp.dok_matrix((self.num_groups, self.num_users), dtype=np.float32)
        for group in gu:
            for user in gu[group]:
                gu_mat[group, user]=1
        return gu_mat

    def create_adj_mat(self, R):
        t1 = time()
        R_shape = list(R.shape)
        row_size = R_shape[0]
        col_size = R_shape[1]
        adj_mat = sp.dok_matrix((row_size,col_size),dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()
        adj_mat = R
        adj_mat = adj_mat.todok()
        #print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_mat_inv = sp.diags(r_inv)
            norm_adj = r_mat_inv.dot(adj)

            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat) #sp.eye(adj_mat.shape[0]) 일단 제외
        mean_adj_mat = normalized_adj_single(adj_mat)

        return norm_adj_mat
        #return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line = line.replace(" ", "")
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_rating_file_as_dict(self, filename):
        ratingList = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line = line.replace(" ", "")
                arr = line.split("\t")
                key = int(arr[0])
                value = int(arr[1])
                if key not in ratingList:
                    ratingList[key] = [value]
                else:
                    ratingList[key]+= [value]
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line = line.replace(" ", "")
                arr = line.split("\t")
                negatives = []
                for x in arr[1:len(arr)-1]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line = line.replace(" ", "")
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                line = line.replace(" ", "")
                arr = line.split("\t")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        #print(mat)
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        *_, last = user_train_loader

        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return group_train_loader

