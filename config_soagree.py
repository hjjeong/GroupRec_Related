'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

class Config():
    def __init__(self):
        self.path = 'data/douban/' #'data/meetup_ca/' #'data/gwl/'
        self.dataset= 'douban' #'meetup.ca' #'gwl'
        self.user_dataset = self.path + 'douban.ui' #'meetup.ca.ui'
        self.group_dataset = self.path + self.dataset
        self.user_in_group_path = self.path+ 'douban.train.gu' #"meetup.ca.train.gu"
        self.embedding_size = 32
        self.epoch = 128
        self.num_negatives = [4]
        self.batch_size = 256
        self.lr =  [0.005, 0.000001, 0.0000005]
        self.drop_ratio = 0.1
        self.topK = [5,10,15,20,25]
        self.layer_size = [32,32,32]
        self.test_flag = ''
        print(self.dataset, self.epoch, self.batch_size)