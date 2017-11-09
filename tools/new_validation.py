from __future__ import division, unicode_literals

import logging
import numpy as np
from tools.new_classifier import MLP, LogReg

import sklearn
assert(sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"


class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, train_loader, dev_loader, test_loader, config):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.nclasses = config['nclasses']
        self.featdim = config["dimension"]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else \
            'pytorch-' + config['classifier']
        self.nepoches = None if 'nepoches' not in config else \
              config['nepoches']
        self.maxepoch = None if 'maxepoch' not in config else \
            config['maxepoch']
        self.noreg = False if 'noreg' not in config else config['noreg']

    def run(self):
        print(self.noreg)
        logging.info('Training {0} with standard validation..'
                     .format(self.modelname))
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        if self.noreg:
            regs = [0.]
        scores = []
        for reg in regs:
            clf = self.create_classifier(reg)
            clf.fit(train_dataloader=self.train_loader, dev_dataloader=self.dev_loader)
            scores.append(round(100*clf.score(self.dev_loader), 2))
        logging.info([('reg:'+str(regs[idx]), scores[idx])
                      for idx in range(len(scores))])

        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Validation : best param found is reg = {0} with score \
            {1}'.format(optreg, devaccuracy))

        clf = self.create_classifier(optreg)
        clf.fit(train_dataloader=self.train_loader, dev_dataloader=self.dev_loader)
        testaccuracy = clf.score(self.test_loader)
        testaccuracy = round(100*testaccuracy, 2)
        return devaccuracy, testaccuracy

    def create_classifier(self, reg):
        if self.classifier == 'LogReg':
            clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses,
                         l2reg=reg, seed=self.seed,
                         use_cuda=self.cudaEfficient)
        elif self.classifier == 'MLP':
            clf = MLP(inputdim=self.featdim, hiddendim=self.nhid,
                      nclasses=self.nclasses, l2reg=reg,
                      seed=self.seed, use_cuda=self.cudaEfficient)
            # small hack : SNLI specific
        if self.nepoches:
            clf.nepoches = self.nepoches
        if self.maxepoch:
            clf.maxepoch = self.maxepoch

        return clf