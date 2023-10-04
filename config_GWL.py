'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

class Config(object):
    def __init__(self):
        self.path = 'data/gwl/'
        self.dataset = 'gwl'
        self.user_dataset = self.path + 'gwl.ui'
        self.group_dataset = self.path + self.dataset
        self.user_in_group_path = self.path+"gwl.train.gu"
        self.embedding_size = 32
        self.epoch = 50
        self.num_negatives = [4]
        self.batch_size = 256
        self.lr = [0.01, 0.000001, 0.0000005]
        self.drop_ratio = 0.2
        self.topK = [5,10,15,20,25]
        print(self.dataset, self.epoch, self.batch_size, self.lr)