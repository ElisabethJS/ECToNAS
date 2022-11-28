'''
Configuration information for tensorflow.keras.models
'''

from __future__ import annotations
from util.layer_analysis import LayerAnalysis
from util.layer_type import ConvolutionalBlock

__version__ = '3.0'
__author__ = 'EJ Schiessler'

# pylint: disable=too-few-public-methods

class ModelAnalysis():
    '''
    Configuration information for tensorflow.keras.models
    '''
    model_config : dict
    convolutional_blocks : dict[int, ConvolutionalBlock]
    flatten_layer : int
    layer_analyses : dict[int, LayerAnalysis]
    layer_count : int
    name : str
    pooling_layer_count : int

    def __init__(self):
        self.model_config = {}
        self.convolutional_blocks = {}
        self.flatten_layer = -1
        self.layer_analyses = {}
        self.layer_count = 0
        self.name = ''
        self.pooling_layer_count = 0
