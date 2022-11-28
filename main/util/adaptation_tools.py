'''
Tools for running ECToNAS and keeping track of applied procedures
'''

from __future__ import annotations
import numpy as np
from tensorflow import keras
from util.adaptation_type import AdaptationType
from util.layer_type import LayerType, ConvolutionalBlock
from util.model_analysis import ModelAnalysis
from util.layer_analysis import LayerAnalysis

__version__ = '3.0'
__author__ = 'EJ Schiessler'

# pylint:disable = too-few-public-methods, line-too-long

class ArchitectureAnalysis():
    '''
    Class that analyses the provided keras model and stores information required to
    perform allowed network modifications
    '''

    __allowed_ops_at_layer = {
        LayerType.ACTIVATION: [AdaptationType.REMOVE_CONV_BLOCK],
        LayerType.AVGPOOL2D: [AdaptationType.REMOVE_CONV_BLOCK],
        LayerType.BATCH_NORM: [AdaptationType.REMOVE_CONV_BLOCK],
        LayerType.CONV2D: [AdaptationType.ADD_CHANNELS, AdaptationType.REMOVE_CHANNELS,
                           AdaptationType.REMOVE_CONV_BLOCK],
        LayerType.DENSE: [AdaptationType.ADD_UNITS, AdaptationType.REMOVE_UNITS,
                          AdaptationType.REMOVE_FC],
        LayerType.FLATTEN: [],
        LayerType.INPUT: [],
        LayerType.MAXPOOL2D: [AdaptationType.REMOVE_CONV_BLOCK],
        LayerType.RESHAPE: []
    }

    __allowed_ops_before_layer = {
        LayerType.ACTIVATION: [],
        LayerType.AVGPOOL2D: [],
        LayerType.BATCH_NORM: [],
        LayerType.CONV2D: [AdaptationType.ADD_CONV_BLOCK],
        LayerType.DENSE: [AdaptationType.ADD_FC],
        LayerType.FLATTEN: [AdaptationType.ADD_CONV_BLOCK],
        LayerType.INPUT: [],
        LayerType.MAXPOOL2D: [],
        LayerType.RESHAPE: []
    }

    __class_names_by_layer_type = {
        'Activation': LayerType.ACTIVATION,
        'AveragePooling2D': LayerType.AVGPOOL2D,
        'BatchNormalization': LayerType.BATCH_NORM,
        'Conv2D': LayerType.CONV2D,
        'Dense': LayerType.DENSE,
        'Flatten': LayerType.FLATTEN,
        'InputLayer': LayerType.INPUT,
        'MaxPooling2D': LayerType.MAXPOOL2D,
        'Reshape': LayerType.RESHAPE
    }

    def __init__(self) -> None:
        pass

    def run(self, model:keras.models) -> ModelAnalysis:
        '''
        Analyses the stored model
        '''
        model_analysis = ModelAnalysis()
        model_analysis.model_config = model.get_config()
        model_analysis.name = model.name
        model_analysis.layer_count = len(model.layers)
        layer_index = 0
        config_index = 1 #discrepancy because keras model configs always include input layer, layer list never does
        convolution_block_counter = -1
        add_conv_blocks_allowed = True
        for layer in model.layers:
            layer_analysis = LayerAnalysis()
            layer_config = model_analysis.model_config['layers'][config_index]

            layer_class_name = layer_config['class_name']
            layer_analysis.layer_type = self.__parse_layer_class_name(layer_class_name)
            layer_analysis.name = layer_config['config']['name']
            layer_analysis.weights = layer.get_weights()
            layer_analysis.weight_shapes = [weight.shape for weight in layer_analysis.weights]
            layer_analysis.unit_count = layer_config['config'].get('units', 0)
            layer_analysis.use_bias = (len(layer_analysis.weight_shapes) > 1)
            layer_analysis.channel_count = layer_config['config'].get('filters', 0)
            layer_analysis.input_shape = layer.input_shape

            # Convolutional blocks
            if layer_analysis.layer_type == LayerType.CONV2D:
                convolution_block_counter += 1
                conv_block = ConvolutionalBlock(convolution_block_counter)
                conv_block.starts_at_layer = layer_index
                conv_block.involved_layers.append(layer_index)
                conv_block.involved_layer_types.append(layer_analysis.layer_type)
                model_analysis.convolutional_blocks[convolution_block_counter] = conv_block
                layer_analysis.belongs_to_conv_block = convolution_block_counter
            elif layer_analysis.layer_type in [LayerType.AVGPOOL2D, LayerType.MAXPOOL2D, LayerType.BATCH_NORM, LayerType.ACTIVATION]:
                conv_block = model_analysis.convolutional_blocks.get(convolution_block_counter, None)
                if conv_block is not None:
                    conv_block.involved_layers.append(layer_index)
                    conv_block.involved_layer_types.append(layer_analysis.layer_type)
                    layer_analysis.belongs_to_conv_block = conv_block.block_number
                if layer_analysis.layer_type in [LayerType.AVGPOOL2D, LayerType.MAXPOOL2D]:
                    model_analysis.pooling_layer_count += 1

            # allowed ops read out from dictionary & perform checks
            layer_analysis.allowed_ops_at = self.__allowed_ops_at_layer.get(layer_analysis.layer_type, [])
            layer_analysis.allowed_ops_before = self.__allowed_ops_before_layer.get(layer_analysis.layer_type, [])
            self.__handle_degenerate_layer(layer_analysis)

            # identify first flatten layer
            if model_analysis.flatten_layer == -1 and layer_analysis.layer_type == LayerType.FLATTEN:
                model_analysis.flatten_layer = layer_index
                if 1 in [layer_analysis.input_shape[0], layer_analysis.input_shape[1]]:
                    add_conv_blocks_allowed = False
            # Dense layer after flatten layer is restricted
            if layer_index == model_analysis.flatten_layer + 1 and layer_analysis.layer_type == LayerType.DENSE:
                layer_analysis.allowed_ops_before = []
            model_analysis.layer_analyses[layer_index] = layer_analysis
            layer_index += 1
            config_index += 1

        # change allowed ops of last layer
        last_layer_config = model_analysis.layer_analyses[layer_index - 1]
        last_layer_config.allowed_ops_at = []
        # add conv block not allowed if flatten layer has 1 in first two dimensions of input shape
        if not add_conv_blocks_allowed:
            for layer_analysis in model_analysis.layer_analyses.values():
                layer_analysis.allowed_ops_before = [op for op in layer_analysis.allowed_ops_before if not op==AdaptationType.ADD_CONV_BLOCK]

        return model_analysis

    def __handle_degenerate_layer(self, layer_config:LayerAnalysis) -> None:
        '''
        Checks if the layer is degenerate. If so, the only allowed adaptation type
        for this layer will be to remove it.
        '''
        if layer_config.layer_type not in [LayerType.DENSE, LayerType.CONV2D]:
            return
        if np.isnan(layer_config.weights[0]).any() or (layer_config.unit_count == 0 and layer_config.channel_count == 0):
            if layer_config.layer_type == LayerType.DENSE:
                layer_config.allowed_ops_at = [AdaptationType.REMOVE_FC]
            else:
                layer_config.allowed_ops_at = [AdaptationType.REMOVE_CONVOLUTION, AdaptationType.REMOVE_CONV_BLOCK]
            layer_config.allowed_ops_before = []

    def __parse_layer_class_name(self, layer_class_name:str) -> LayerType:
        '''
        Translates the layer_class_name into a LayerType
        '''
        return self.__class_names_by_layer_type.get(layer_class_name, LayerType.OTHER)

class ModificationReport():
    '''
    Class that contains information about the model and procedures that were applied to it
    '''
    adaptation_type: AdaptationType
    adapted_model: keras.models
    description: str

    def __init__(self, model:keras.models, description:str, adaptation_type:AdaptationType) -> None:
        self.adaptation_type = adaptation_type
        self.adapted_model = model
        self.description = description

class ModificationInstruction():
    '''
    Class that contains instruction details on what adaptations to perform
    '''
    activation: str
    adaptation_type: AdaptationType
    adaptation_steps: list[AdaptationType]
    add_before_layer: int
    add_to_layer: int
    description: str
    architecture_analysis: ModelAnalysis
    remove_from_layer: int
    remove_layers: list[int]
    target_units_filters_count: int
    use_bias: bool

    def __init__(self, model_analysis:ModelAnalysis) -> None:
        self.architecture_analysis = model_analysis
        self.reset()

    def reset(self) -> None:
        '''
        Resets the modification instructions
        '''
        self.activation = 'relu'
        self.adaptation_type = None
        self.adaptation_steps = []
        self.add_before_layer = None
        self.add_to_layer = None
        self.description = ''
        self.remove_from_layer = None
        self.remove_layers = []
        self.target_units_filters_count = 0
        self.use_bias = True
