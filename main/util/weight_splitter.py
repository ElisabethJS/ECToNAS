'''
Stores information on model weights relative to layers
that are affected by a model adaptation
'''

from __future__ import annotations
import numpy as np

__version__ = '1.0'
__author__ = 'EJ Schiessler'

#pylint: disable=too-few-public-methods

class WeightSplitter():
    '''
    Stores information on model weights relative to layers
    that are affected by a model adaptation
    '''
    before : list[np.ndarray]
    main : list[np.ndarray]
    intermediary_1 : list[np.ndarray]
    bn_secondary : list[np.ndarray]
    intermediary_2 : list[np.ndarray]
    secondary : list[np.ndarray]
    after : list[np.ndarray]

    def __init__(self):
        self.before = []
        self.main = []
        self.intermediary_1 = []
        self.bn_secondary = []
        self.intermediary_2 = []
        self.secondary = []
        self.after = []
