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

import logging

from tools.new_validation import SplitClassifier


class BinaryClassifierEval(object):
    def __init__(self, train, dev, test, seed=1111):
        self.seed = seed
        self.train = train
        self.dev = dev
        self.test = test

    def run(self, params):

        logging.info('Generated sentence embeddings')

        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params['usepytorch'],
                             'classifier': params['classifier'],
                             'dimension': params['dimension'],
                             'cudaEfficient': params['cudaEfficient'],
                             'nhid': params["dimension"],
                             "nepoches": params["nepoches"],
                             "maxepoch": params["maxepoch"]}

        clf = SplitClassifier(train_loader=self.train, test_loader=self.test,
                              dev_loader=self.dev, config=config_classifier)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'dev_acc': devacc, 'test_acc': testacc}

