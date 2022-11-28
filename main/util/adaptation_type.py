'''
Adaptation types that can be handled by ECToNAS and related code
'''

from enum import Enum

__version__ = '3.0'
__author__ = 'EJ Schiessler'

class AdaptationType(Enum):
    '''
    Adaptation types that can be handled by ECToNAS and related code
    '''
    UNSPECIFIED = -1
    IDENTITY = 0
    ADD_ACTIVATION = 9
    ADD_BATCH_NORMALIZTATION = 10
    ADD_CHANNELS = 1
    ADD_CONV_BLOCK = 2
    ADD_CONVOLUTION = 11
    ADD_DIMENSION = 12
    ADD_FC = 3
    ADD_POOL_AVG = 13
    ADD_POOL_MAX = 14
    ADD_UNITS = 4
    REMOVE_ACTIVATION = 15
    REMOVE_BATCH_NORMALIZATION = 16
    REMOVE_CHANNELS = 5
    REMOVE_CONV_BLOCK = 6
    REMOVE_CONVOLUTION = 17
    REMOVE_FC = 7
    REMOVE_POOL = 18
    REMOVE_UNITS = 8
