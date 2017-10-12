# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import division, unicode_literals

import io
import os
import numpy as np
import logging

from tools.validation import KFoldClassifier


class BinaryClassifierEval(object):
    def __init__(self, train, test, seed=1111):
        self.seed = seed
        self.train = train
        self.test = test
        self.n_samples = self.train["X"].shape[0]

    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.samples)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='utf-8') as f:
            return [line.split() for line in f.read().splitlines()]

    def run(self, params):

        logging.info('Generated sentence embeddings')

        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params['usepytorch'],
                             'classifier': params['classifier'],
                             'nhid': self.train["X"].shape[1], 'kfold': params["kfold"]}

        clf = KFoldClassifier(train=self.train, test=self.test, config=config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': self.n_samples,
                'ntest': self.n_samples}

