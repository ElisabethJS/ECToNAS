'''
Layer types that can be handled by ECToNAS and related code
'''

from __future__ import annotations
from enum import Enum

__version__ = '3.0'
__author__ = 'EJ Schiessler'

class LayerType(Enum):
    '''
    Layer types that can be handled by ECToNAS and related code
    '''
    OTHER = -1
    ACTIVATION = 0
    AVGPOOL2D = 1
    BATCH_NORM = 2
    CONV2D = 3
    DENSE = 4
    FLATTEN = 5
    INPUT = 6
    MAXPOOL2D = 7
    RESHAPE = 8

class ConvolutionalBlock():
    '''
    Convolutional block units
    '''
    block_number: int
    involved_layers: list[int]
    involved_layer_types: list[LayerType]
    starts_at_layer: int

    def __init__(self, block_number:int) -> None:
        self.block_number = block_number
        self.involved_layers = []
        self.involved_layer_types = []
        self.starts_at_layer = -1
