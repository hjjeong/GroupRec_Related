'''
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao
'''
import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq
import multiprocessing

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True
        self._group_recall = {}

    def evaluate_model(self, model, testRatings, testNegatives, K):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        #hits1, ndcgs1 = [], []
        #for idx in range(len(testRatings)):
        #    (hr,ndcg) = self.eval_one_rating(model, testRatings, testNegatives, 5, idx) #여기서 대체 왜 죽지
        #    hits1.append(hr)
        #    ndcgs1.append(ndcg)
        #recs1 = []
        #for gidx in self._group_recall:
        #    recs1.append(float(sum(self._group_recall[gidx])) / len(self._group_recall[gidx]))

        hits2, ndcgs2 = [], []
        for idx in range(len(testRatings)):
            (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, 10, idx)
            hits2.append(hr)
            ndcgs2.append(ndcg)
        recs2 = []
        for gidx in self._group_recall:
            recs2.append(float(sum(self._group_recall[gidx])) / len(self._group_recall[gidx]))
        """
        hits3, ndcgs3 = [], []
        for idx in range(len(testRatings)):
            (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, 15, idx)
            hits3.append(hr)
            ndcgs3.append(ndcg)
        recs3 = []
        for gidx in self._group_recall:
            recs3.append(float(sum(self._group_recall[gidx])) / len(self._group_recall[gidx]))

        hits4, ndcgs4 = [], []
        for idx in range(len(testRatings)):
            (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, 20, idx)
            hits4.append(hr)
            ndcgs4.append(ndcg)
        recs4 = []
        for gidx in self._group_recall:
            recs4.append(float(sum(self._group_recall[gidx])) / len(self._group_recall[gidx]))

        hits5, ndcgs5 = [], []
        for idx in range(len(testRatings)):
            (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, 25, idx)
            hits5.append(hr)
            ndcgs5.append(ndcg)
        recs5 = []
        for gidx in self._group_recall:
            recs5.append(float(sum(self._group_recall[gidx])) / len(self._group_recall[gidx]))
        """
        #return (hits1, ndcgs1, recs1, hits2, ndcgs2, recs2, hits3, ndcgs3, recs3, hits4, ndcgs4, recs4, hits5, ndcgs5, recs5)
        return (hits2, ndcgs2, recs2)
    def eval_one_rating(self, model, testRatings, testNegatives, K, idx): #1개 그룹의 모든 item pair(train제외)에 대해 batch 만들어서 test
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)
        users_var = torch.LongTensor(users)
        items_var = torch.LongTensor(items)
        #print(type_m)
        predictions = model(users_var, items_var).cpu()
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()
        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        self.getRecall(u, ranklist, gtItem)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    def getRecall(self,gidx, ranklist, gtItem):
        if gtItem not in ranklist:
            if gidx not in self._group_recall:
                self._group_recall[gidx] = [0]
            else:
                self._group_recall[gidx] += [0]
        else:
            if gidx not in self._group_recall:
                self._group_recall[gidx] = [1]
            else:
                self._group_recall[gidx] += [1]