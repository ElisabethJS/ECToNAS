'''
Configuration information for tensorflow.keras.model.layers
'''

from __future__ import annotations
import numpy as np
from util.layer_type import LayerType
from util.adaptation_type import AdaptationType

__version__ = '3.0'
__author__ = 'EJ Schiessler'

# pylint: disable=too-few-public-methods, too-many-instance-attributes

class LayerAnalysis():
    '''
    Configuration information for tensorflow.keras.model.layers
    '''
    allowed_ops_at: list[AdaptationType]
    allowed_ops_before : list[AdaptationType]
    belongs_to_conv_block: int
    channel_count : int
    input_shape : tuple
    layer_type : LayerType
    name : str
    unit_count : int
    use_bias : bool
    weight_shapes : list[tuple]
    weights : list[np.ndarray]

    def __init__(self):
        self.allowed_ops_at = []
        self.allowed_ops_before = []
        self.belongs_to_conv_block = None
        self.channel_count = 0
        self.input_shape = ()
        self.layer_type = LayerType.OTHER
        self.name = ''
        self.unit_count = 0
        self.use_bias = False
        self.weight_shapes = []
        self.weights = []
