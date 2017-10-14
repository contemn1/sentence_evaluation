# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Modified Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable


class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 use_cuda=True, nepoches=4, maxepoch=200):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.nepoches = nepoches
        self.maxepoch = maxepoch

    def fit(self, train_dataloader, dev_dataloader,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            self.trainepoch(train_dataloader, nepoches=self.nepoches)
            accuracy = self.score(dev_dataloader)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= 5:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy

    def trainepoch(self, dataloader, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            all_costs = []
            for Xbatch, ybatch in dataloader:
                # forward
                if self.use_cuda:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                Xbatch = Variable(Xbatch)
                ybatch = Variable(ybatch)
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data[0])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def score(self, dev_dataloader):
        self.model.eval()
        correct = 0
        for Xbatch, ybatch in dev_dataloader:
            if self.use_cuda:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            Xbatch = Variable(Xbatch, volatile=True)

            output = self.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()
        accuracy = 1.0 * correct / len(dev_dataloader)
        return accuracy

    def predict(self, dev_loader):
        self.model.eval()
        yhat = np.array([])
        for Xbatch, _ in dev_loader:
            if self.use_cuda:
                Xbatch = Xbatch.cuda()
            output = self.model(Xbatch)
            yhat = np.append(yhat,
                             output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, dev_loader):
        self.model.eval()
        probas = []
        for Xbatch, _ in dev_loader:
            Xbatch = Variable(Xbatch)
            if not probas:
                probas = self.model(Xbatch).data.cpu().numpy()
            else:
                probas = np.concatenate(probas,
                                        self.model(Xbatch).data.cpu().numpy(),
                                        axis=0)
        return probas


"""
Logistic Regression with Pytorch
"""


class LogReg(PyTorchClassifier):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, use_cuda=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, use_cuda)
        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.nclasses),
        ).cuda()
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)


"""
MLP with Pytorch
"""


class MLP(PyTorchClassifier):
    def __init__(self, inputdim, hiddendim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, use_cuda=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, use_cuda)

        self.hiddendim = hiddendim

        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.hiddendim),
            # TODO : add parameter p for dropout
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(self.hiddendim, self.nclasses),
            nn.Softmax()
        ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)
