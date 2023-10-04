'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

class Config(object):
    def __init__(self):
        self.path = 'data/meetup_nyc/'
        self.dataset = 'meetup.nyc'
        self.user_dataset = self.path + 'meetup.nyc.ui'
        self.group_dataset = self.path + self.dataset
        self.user_in_group_path = self.path+"meetup.nyc.train.gu"
        self.embedding_size = 32
        self.epoch = 60
        self.num_negatives = [4]
        self.batch_size = 256
        self.lr = [0.01, 0.0005, 0.0001]
        self.drop_ratio = 0.2
        self.topK = [5, 10, 15, 20, 25]
        self.layer_size = [32, 32, 32]
        print(self.group_dataset)