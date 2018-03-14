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
import torch.nn.functional as F



class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 use_cuda=True):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def fit(self, train_dataloader, dev_dataloader,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            loss = self.trainepoch(train_dataloader, nepoches=self.epoch_size)
            accuracy = self.score(dev_dataloader)
            accuracy_train = self.score(train_dataloader)
            print("current epoch is {0}, accuracy is {1}, best accuracy is {2} acc train is {3}".format(self.nepoch, accuracy, bestaccuracy, accuracy_train))

            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)

            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
                print("current epoch is {0}, early stop count is {1}".format(self.nepoch, early_stop_count))

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
        return np.mean(all_costs)

    def score(self, dev_dataloader):
        self.model.eval()
        correct = 0
        number_of_data = 0
        for Xbatch, ybatch in dev_dataloader:
            number_of_data += Xbatch.size()[0]
            if self.use_cuda:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            Xbatch = Variable(Xbatch, volatile=True)
            ybatch = Variable(ybatch, volatile=True)

            output = self.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()
        accuracy = (1.0 * correct) / number_of_data
        return accuracy

    def predict(self, dev_loader):
        self.model.eval()
        yhat = np.array([])
        for Xbatch, _ in dev_loader:
            Xbatch = Variable(Xbatch, volatile=True)
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
            Xbatch = Variable(Xbatch, volatile=True)
            vals = F.softmax(self.model(Xbatch).data.cpu().numpy())

            if not probas:
                probas = vals
            else:
                probas = np.concatenate(probas,
                                        vals,
                                        axis=0)
        return probas


"""
Logistic Regression with Pytorch
"""


class LogReg(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, use_cuda=True):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, use_cuda)
        self.epoch_size = 1 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
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
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, use_cuda=True):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, use_cuda)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
                )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
                )

        if use_cuda:
            self.model = self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)
