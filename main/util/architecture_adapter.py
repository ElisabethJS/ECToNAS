'''
Class that performs architecture adaptations
'''

from __future__ import annotations
import warnings
import re
from copy import deepcopy
import numpy as np
from tensorflow import keras, python
from util.adaptation_type import AdaptationType
from util.layer_type import LayerType
from util.adaptation_tools import ModificationInstruction, ArchitectureAnalysis
from util.weight_splitter import WeightSplitter

__version__ = '3.0'
__author__ = 'EJ Schiessler'

# pylint:disable = line-too-long

class ArchitectureAdapter():
    '''
    Network architecture modifications for FCNNs and CNNs
    '''

    adaptation_type: AdaptationType
    config_index: int
    input_model: keras.models.Sequential
    input_config: dict
    instruction: ModificationInstruction
    output_model: keras.models.Sequential
    output_config: dict
    position_index: int
    weight_splitter: WeightSplitter

    _layer_name_by_type = {
        LayerType.ACTIVATION: 'activation',
        LayerType.AVGPOOL2D: 'average_pooling2d',
        LayerType.BATCH_NORM: 'batch_normalization',
        LayerType.CONV2D: 'conv2d',
        LayerType.DENSE: 'dense',
        LayerType.FLATTEN: 'flatten',
        LayerType.MAXPOOL2D: 'max_pooling2d',
        LayerType.RESHAPE: 'reshape'
    }

    def __init__(self) -> None:
        self.config_index = -1
        self.input_model = None
        self.input_config = None
        self.instruction = None
        self.output_model = None
        self.output_config = None
        self.position_index = -1
        self.weight_splitter = None

    def Run(self, model:keras.models.Sequential, instruction:ModificationInstruction) -> keras.models.Sequential:
        '''
        Performs architectural changes on the given model
        '''
        self.output_model = model
        self.instruction = instruction
        try:
            for adaptation_type in self.instruction.adaptation_steps:
                self.adaptation_type = adaptation_type
                self.__prepare()
                self.__split_weights()
                self.__modify()
                self.__build_model()
        except MemoryError:
            warnings.warn(f'Memory error when performing {adaptation_type}')
            return None
        return self.output_model

    def __add_bn_channels(self, bn_weights:list[np.ndarray], channels_to_add:int) -> list[np.ndarray]:
        '''
        Adds channels to a batch normalization layer
        '''
        if len(bn_weights) > 0:
            ones = np.ones(channels_to_add, np.float32)
            for idx in [0, 3]:
                bn_weights[idx] = np.append(bn_weights[idx], ones, axis=0)
            zeros = np.zeros(channels_to_add, np.float32)
            for idx in [1, 2]:
                bn_weights[idx] = np.append(bn_weights[idx], zeros, axis=0)
        return bn_weights

    def __add_conv_channels(self, channels_to_add:int):
        '''
        Adds channels to a convolutional layer and adapts next bn and
        next layer with weights
        '''
        main_weights = self.weight_splitter.main
        bn_weights = self.weight_splitter.bn_secondary
        secondary_weights = self.weight_splitter.secondary
        current_shape = main_weights[0].shape
        full_loops = channels_to_add // current_shape[-1]
        loop_remainder = channels_to_add % current_shape[-1]
        flatten_layer_inbetween = False
        if len(secondary_weights[0].shape) <= 2 < len(current_shape):
            flatten_layer_inbetween = True
            flatten_input_shape = self.instruction.architecture_analysis.layer_analyses[self.instruction.architecture_analysis.flatten_layer].input_shape

        for idx, value in enumerate(loop_remainder*[full_loops + 1] + (current_shape[-1]-loop_remainder)*[full_loops]):
            if value > 0:
                if flatten_layer_inbetween:
                    secondary_weights[0] = secondary_weights[0].reshape(flatten_input_shape[1:] + (-1,))

                main_weights[0][..., idx] = main_weights[0][..., idx]/(value + 1)
                for _ in range(value):
                    main_weights[0] = np.append(main_weights[0], main_weights[0][..., idx, None], axis=-1)
                    secondary_weights[0] = np.append(secondary_weights[0], secondary_weights[0][..., [idx], :], axis=-2)

                if flatten_layer_inbetween:
                    secondary_weights[0] = secondary_weights[0].reshape((-1,) + (secondary_weights[0].shape[-1],))
                    flatten_input_shape = flatten_input_shape[:-1] + (flatten_input_shape[-1] + value,)

                if len(main_weights) > 1:
                    main_weights[1][idx] = main_weights[1][idx]/(value+1)
                    for _ in range(value):
                        main_weights[1] = np.append(main_weights[1], [main_weights[1][idx]], axis=0)

        bn_weights = self.__add_bn_channels(bn_weights, channels_to_add)
        self.weight_splitter.main = main_weights
        self.weight_splitter.bn_secondary = bn_weights
        self.weight_splitter.secondary = secondary_weights

    def __build_model(self) -> None:
        '''
        Builds a keras model from the provided config and weights list
        '''
        self.output_model = keras.Sequential.from_config(self.output_config)
        modified_weights = self.weight_splitter.before + self.weight_splitter.main + self.weight_splitter.intermediary_1
        modified_weights += self.weight_splitter.bn_secondary + self.weight_splitter.intermediary_2
        modified_weights += self.weight_splitter.secondary + self.weight_splitter.after
        self.output_model.set_weights(modified_weights)

    def __determine_position_index(self) -> None:
        '''
        Checks the surgery instructions and extracts the affected layer position
        '''
        position_index = self.instruction.add_before_layer or self.instruction.add_to_layer or self.instruction.remove_from_layer
        if position_index is not None:
            self.position_index = position_index
            return
        if len(self.instruction.remove_layers) > 0:
            self.position_index = self.instruction.remove_layers[0]
            return
        self.position_index = 0

    def __get_layer_name(self, layer_type:LayerType) -> str:
        '''
        Gets the appropriate layer name for the given layer type that is
        unique for the current model
        '''
        name_str = self._layer_name_by_type.get(layer_type, 'unknown_type')
        highest_cnt = -1
        for layer_analysis in self.instruction.architecture_analysis.layer_analyses.values():
            if re.search(f'^{name_str}', layer_analysis.name) is not None:
                ints_in_name = re.findall(r'\d+', layer_analysis.name)
                if len(ints_in_name) > 0:
                    highest_cnt = np.maximum(highest_cnt, int(ints_in_name[-1]))

        return f'{name_str}_{highest_cnt + 1}'

    def __get_next_layer_with_weights(self, start_from_layer_index:int) -> list[int]:
        '''
        Goes through the layer list and returns the index of the next layer that has
        weights.
        If the next layer happens to be a batch normalization layer, the following layer
        after that with non-zero weights is also returned.
        '''
        layers_with_weights = []
        for index in sorted(self.instruction.architecture_analysis.layer_analyses.keys())[start_from_layer_index + 1:]:
            layer = self.instruction.architecture_analysis.layer_analyses[index]
            if len(layer.weights) > 0:
                layers_with_weights.append(index)
                if not layer.layer_type == LayerType.BATCH_NORM:
                    return layers_with_weights
        return layers_with_weights

    def __is_pooling_remove_full_sized(self) -> list[bool, bool]:
        '''
        When removing a pooling layer we need to check if the input dimension to the flatten layer gets
        doubled by this operation or not
        '''
        first_input_shape = self.instruction.architecture_analysis.layer_analyses[0].input_shape
        pooling_count = self.instruction.architecture_analysis.pooling_layer_count
        non_full_divisions = [[], []]
        result = [True, True]
        for dim_cnt in range(2):
            shape = first_input_shape[dim_cnt+1]
            for layer_cnt in range(pooling_count):
                shape = shape/2
                if not shape.is_integer():
                    non_full_divisions[dim_cnt].append(layer_cnt + 1)
                    shape += 0.5
            if len(non_full_divisions[dim_cnt]) > 0 and non_full_divisions[dim_cnt][-1] == pooling_count:
                result[dim_cnt] = False
        return result

    def __modify(self) -> None:
        '''
        Pass through to modification function by adaptation type
        '''
        modification_function_by_adaptation_type = {
            AdaptationType.IDENTITY: self.__modify_identity,
            AdaptationType.ADD_ACTIVATION: self.__modify_add_activation,
            AdaptationType.ADD_BATCH_NORMALIZTATION: self.__modify_add_batch_normalization,
            AdaptationType.ADD_CHANNELS: self.__modify_add_channels,
            AdaptationType.ADD_CONVOLUTION: self.__modify_add_convolution,
            AdaptationType.ADD_DIMENSION: self.__modify_add_dimension,
            AdaptationType.ADD_FC: self.__modify_add_fc,
            AdaptationType.ADD_POOL_AVG: self.__modify_add_pooling,
            AdaptationType.ADD_POOL_MAX: self.__modify_add_pooling,
            AdaptationType.ADD_UNITS: self.__modify_add_units,
            AdaptationType.REMOVE_ACTIVATION: self.__modify_remove_activation,
            AdaptationType.REMOVE_BATCH_NORMALIZATION: self.__modify_remove_batch_normalization,
            AdaptationType.REMOVE_CHANNELS: self.__modify_remove_channels,
            AdaptationType.REMOVE_CONVOLUTION: self.__modify_remove_convolution,
            AdaptationType.REMOVE_FC: self.__modify_remove_fc,
            AdaptationType.REMOVE_POOL: self.__modify_remove_pooling,
            AdaptationType.REMOVE_UNITS: self.__modify_remove_units
        }
        self.output_config = deepcopy(self.input_config)
        modification_function_by_adaptation_type[self.adaptation_type]()
        #modification_function_by_adaptation_type[AdaptationType.IDENTITY]()

    def __prepare(self) -> None:
        '''
        Preparations for running ECToNAS
        '''
        self.input_model = keras.models.clone_model(self.output_model)
        self.input_model.set_weights(self.output_model.get_weights())
        self.input_config = self.input_model.get_config()
        self.instruction.architecture_analysis = ArchitectureAnalysis().run(self.input_model)
        self.output_model = None
        self.output_config = None
        self.weight_splitter = WeightSplitter()
        self.__determine_position_index()
        self.config_index = self.position_index + 1

    def __remove_bn_channels(self, bn_weights:list[np.ndarray], channels_to_remove:list[int]) -> list[np.ndarray]:
        '''
        Removes the specified batch normalization channels
        Returns the updated batch normalization weights
        '''
        if len(bn_weights) == 0:
            return bn_weights
        secondary_weights = self.weight_splitter.secondary
        if len(secondary_weights) > 1:
            for i in channels_to_remove:
                secondary_weights[1] += bn_weights[1][i]
        self.weight_splitter.secondary = secondary_weights
        keep_indices = [i for i in range(len(bn_weights[0])) if i not in channels_to_remove]
        new_bn_weights = []
        if len(keep_indices) > 0:
            for weight in bn_weights:
                new_bn_weights.append(weight[keep_indices])
        return new_bn_weights

    def __remove_conv_channels(self, number_to_remove:int):
        '''
        Removes channels from a convolutional layer and the affected batch normalization
        and follow up layers
        '''
        current_channels = self.weight_splitter.main[0].shape[-1]
        bn_weights = self.weight_splitter.bn_secondary
        channels_to_remove = np.arange(current_channels).tolist()
        if len(bn_weights) > 0:
            _, channels_to_remove = zip(*sorted(zip(bn_weights[0].tolist(), channels_to_remove)))
        channels_to_remove = channels_to_remove[:number_to_remove]
        channels_to_remove = sorted(channels_to_remove)
        self.weight_splitter.bn_secondary = self.__remove_bn_channels(bn_weights, channels_to_remove)
        self.weight_splitter.main[0] = np.delete(self.weight_splitter.main[0], channels_to_remove, 3)
        if len(self.weight_splitter.main) > 1:
            self.weight_splitter.main[1] = np.delete(self.weight_splitter.main[1], channels_to_remove, 0)

        secondary_weights = self.weight_splitter.secondary
        flatten_layer_inbetween = False
        if len(secondary_weights[0].shape) <= 2:
            flatten_layer_inbetween = True
            flatten_input_shape = self.instruction.architecture_analysis.layer_analyses[self.instruction.architecture_analysis.flatten_layer].input_shape
            secondary_weights[0] = secondary_weights[0].reshape(flatten_input_shape[1:] + (-1,))
        secondary_weights[0] = np.delete(secondary_weights[0], channels_to_remove, 2)
        if flatten_layer_inbetween:
            secondary_weights[0] = secondary_weights[0].reshape((-1,) + (secondary_weights[0].shape[-1],))
        self.weight_splitter.secondary = secondary_weights

    def __split_weights(self) -> None:
        '''
        Sorts the model's weights into categories such that the correct weights are updated
        '''
        before = np.arange(self.position_index).tolist()
        main = [self.position_index]
        bn_secondary = None
        intermediary_1 = []
        secondary = None
        intermediary_2 = []
        after = []

        layer_count = self.instruction.architecture_analysis.layer_count
        # no layer weights affected or only main layer:
        if self.adaptation_type in [AdaptationType.IDENTITY, AdaptationType.ADD_ACTIVATION, AdaptationType.REMOVE_ACTIVATION,
                                    AdaptationType.ADD_DIMENSION, AdaptationType.ADD_FC, AdaptationType.ADD_CONVOLUTION,
                                    AdaptationType.ADD_BATCH_NORMALIZTATION]:
            pass

        # main and next with weights affected
        elif self.adaptation_type in [AdaptationType.ADD_UNITS, AdaptationType.REMOVE_UNITS, AdaptationType.REMOVE_FC,
                                      AdaptationType.REMOVE_BATCH_NORMALIZATION]:
            secondary = self.__get_next_layer_with_weights(self.position_index)[0]

        # layer after flatten layer affected:
        elif self.adaptation_type in [AdaptationType.ADD_POOL_AVG, AdaptationType.ADD_POOL_MAX, AdaptationType.REMOVE_POOL]:
            secondary = self.__get_next_layer_with_weights(self.instruction.architecture_analysis.flatten_layer)[0]

        # next bn layer, next layer with weights
        # nb: we run into problems if more than one bn layer follows before the next with weights, but that should not happen
        # anyways (at least in theory..)
        elif self.adaptation_type in [AdaptationType.ADD_CHANNELS, AdaptationType.REMOVE_CHANNELS, AdaptationType.REMOVE_CONVOLUTION]:
            secondary = self.__get_next_layer_with_weights(self.position_index)
            if len(secondary) > 1:
                bn_secondary = secondary[0]
            secondary = secondary[-1]

        if secondary is None:
            after = np.arange(self.position_index+1, layer_count).tolist()
            bn_secondary = []
            secondary = []
        else:
            after = np.arange(secondary + 1, layer_count).tolist()
            if bn_secondary is None:
                intermediary_1 = np.arange(self.position_index + 1, secondary).tolist()
                bn_secondary = []
            else:
                intermediary_1 = np.arange(self.position_index + 1, bn_secondary).tolist()
                intermediary_2 = np.arange(bn_secondary + 1, secondary).tolist()
                bn_secondary = [bn_secondary]
            secondary = [secondary]

        weight_assignements = [
            (self.weight_splitter.before, before),
            (self.weight_splitter.main, main),
            (self.weight_splitter.intermediary_1, intermediary_1),
            (self.weight_splitter.bn_secondary, bn_secondary),
            (self.weight_splitter.intermediary_2, intermediary_2),
            (self.weight_splitter.secondary, secondary),
            (self.weight_splitter.after, after)
        ]

        for (assignement, idxs) in weight_assignements:
            for idx in idxs:
                if idx < layer_count:
                    assignement += self.instruction.architecture_analysis.layer_analyses[idx].weights

    def __modify_add_activation(self) -> None:
        '''
        Adds an activation function
        '''
        activation_config = {'class_name': 'Activation',
                             'config': {'name': self.__get_layer_name(LayerType.ACTIVATION),
                                        'trainable': True,
                                        'dtype': 'float32',
                                        'activation': self.instruction.activation}}
        self.output_config['layers'].insert(self.config_index, activation_config)

    def __modify_add_batch_normalization(self) -> None:
        '''
        Adds a batch normalization layer
        '''
        batch_normalization_config = {'class_name': 'BatchNormalization',
                                      'config': {'name': self.__get_layer_name(LayerType.BATCH_NORM),
                                                'trainable': True,
                                                'dtype': 'float32',
                                                'axis': python.training.tracking.data_structures.ListWrapper([3]),
                                                'momentum': 0.99,
                                                'epsilon': 0.001,
                                                'center': True,
                                                'scale': True,
                                                'beta_initializer': {'class_name': 'Zeros', 'config': {}},
                                                'gamma_initializer': {'class_name': 'Ones', 'config': {}},
                                                'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
                                                'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
                                                'beta_regularizer': None,
                                                'gamma_regularizer': None,
                                                'beta_constraint': None,
                                                'gamma_constraint': None}}
        self.output_config['layers'].insert(self.config_index, batch_normalization_config)
        next_input_shape = self.instruction.architecture_analysis.layer_analyses[self.position_index].input_shape
        zeros = np.zeros(next_input_shape[-1])
        ones = np.ones(next_input_shape[-1])
        bn_weights = [ones, zeros, zeros, ones]
        self.weight_splitter.main = bn_weights + self.weight_splitter.main

    def __modify_add_channels(self) -> None:
        '''
        Adds channels to a convolutional layer
        '''
        channel_target_count = self.instruction.target_units_filters_count
        self.output_config['layers'][self.config_index]['config']['filters'] = channel_target_count
        current_shape = self.weight_splitter.main[0].shape
        channels_to_add = channel_target_count - current_shape[-1]
        self.__add_conv_channels(channels_to_add)

    def __modify_add_convolution(self) -> None:
        '''
        Adds a convolutional layer
        '''
        input_shape = self.input_model.layers[self.position_index].input_shape
        kernel_size = (3, 3)
        conv_config = {'class_name': 'Conv2D',
                       'config': {'name': self.__get_layer_name(LayerType.CONV2D),
                            'trainable': True,
                            'dtype': 'float32',
                            'filters': input_shape[-1],
                            'kernel_size': kernel_size,
                            'strides': (1, 1),
                            'padding': 'same',
                            'data_format': 'channels_last',
                            'dilation_rate': (1, 1),
                            'groups': 1,
                            'activation': 'linear',
                            'use_bias': self.instruction.use_bias,
                            'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
                            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
                            'kernel_regularizer': None,
                            'bias_regularizer': None,
                            'activity_regularizer': None,
                            'kernel_constraint': None,
                            'bias_constraint': None}}

        self.output_config['layers'].insert(self.config_index, conv_config)
        target_shape = (kernel_size[0], kernel_size[1], input_shape[-1], input_shape[-1])
        cnn_weights = [np.zeros(target_shape)]
        cnn_weights[0][(target_shape[0]-1)//2, (target_shape[1]-1)//2, ...] = 1
        if self.instruction.use_bias:
            cnn_weights.append(np.zeros((input_shape[-1],)))
        self.weight_splitter.main = cnn_weights + self.weight_splitter.main

    def __modify_add_dimension(self) -> None:
        '''
        If required, adds a reshape layer. Otherwise this function just returns the model as is
        '''
        next_input_shape = self.instruction.architecture_analysis.layer_analyses[self.position_index].input_shape
        if len(next_input_shape) < 4:
            reshape_config = {
                'class_name': 'Reshape',
                    'config': {'name': self.__get_layer_name(LayerType.RESHAPE),
                               'trainable': True,
                               'batch_input_shape': next_input_shape,
                               'dtype': 'float32',
                               'target_shape': next_input_shape[1:] + (1,)
                              }
            }
            self.output_config['layers'].insert(self.config_index, reshape_config)
            self.instruction.add_before_layer += 1

    def __modify_add_fc(self) -> None:
        '''
        Adds a fully connected layer
        '''
        layer_copy = deepcopy(self.output_config['layers'][self.config_index])
        input_shape = self.input_model.layers[self.position_index].input_shape
        layer_config = {
            'activation': self.instruction.activation,
            'units': input_shape[1],
            'kernel_initializer': {'class_name': 'Identity', 'config': {'gain': 1.0}},
            'use_bias': self.instruction.use_bias,
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'name': self.__get_layer_name(LayerType.DENSE)
        }
        for key, value in layer_config.items():
            layer_copy['config'][key] = value
        self.output_config['layers'].insert(self.config_index, layer_copy)

        additional_weights = [np.identity(input_shape[1])]
        if self.instruction.use_bias:
            additional_weights.append(np.zeros((input_shape[1],)))
        self.weight_splitter.main = additional_weights + self.weight_splitter.main

    def __modify_add_pooling(self) -> None:
        '''
        Adds a pooling layer
        '''
        if self.adaptation_type == AdaptationType.ADD_POOL_AVG:
            layer_class_name = 'AveragePooling2D'
            layer_type = LayerType.AVGPOOL2D
            pooling_fn = np.nanmean
        else:
            layer_class_name = 'MaxPooling2D'
            layer_type = LayerType.MAXPOOL2D
            pooling_fn = np.nanmax
        pooling_config = {'class_name': layer_class_name,
                          'config': {'name': self.__get_layer_name(layer_type),
                                     'trainable': True,
                                     'dtype': 'float32',
                                     'pool_size': (2, 2),
                                     'padding': 'same',
                                     'strides': (2, 2),
                                     'data_format': 'channels_last'
                                  }
        }
        self.output_config['layers'].insert(self.config_index, pooling_config)

        secondary_weights = self.weight_splitter.secondary
        input_shape = self.instruction.architecture_analysis.layer_analyses[self.instruction.architecture_analysis.flatten_layer].input_shape[1:]
        target_shape = input_shape + (secondary_weights[0].shape[-1],)
        secondary_weights[0] = np.reshape(secondary_weights[0], (target_shape[0]*target_shape[1],) + target_shape[2:])
        secondary_weights[0] = np.reshape(secondary_weights[0], target_shape)
        target_shape = (int(np.ceil(target_shape[0]/2)), int(np.ceil(target_shape[1]/2))) + target_shape[2:]
        secondary_new = np.zeros(target_shape)
        for row_idx in range(target_shape[0]):
            for col_idx in range(target_shape[1]):
                secondary_new[row_idx, col_idx, :] = pooling_fn(secondary_weights[0][2*row_idx:2*row_idx+2, 2*col_idx:2*col_idx+2, :, :], axis=(0,1))
        secondary_new = np.reshape(secondary_new, (-1, secondary_new.shape[-1]))
        self.weight_splitter.secondary = [secondary_new]
        if len(secondary_weights) > 1:
            self.weight_splitter.secondary.append(secondary_weights[1])

    def __modify_add_units(self) -> None:
        '''
        Adds units to a fully connected layer
        '''
        self.output_config['layers'][self.config_index]['config']['units'] = self.instruction.target_units_filters_count
        main_weights = self.weight_splitter.main
        secondary_weights = self.weight_splitter.secondary
        current_shape = main_weights[0].shape
        units_to_add =  self.instruction.target_units_filters_count - current_shape[-1]
        full_loops = units_to_add // current_shape[-1]
        loop_remainder = units_to_add % current_shape[-1]

        for idx, value in enumerate(loop_remainder*[full_loops + 1] + (current_shape[-1]-loop_remainder)*[full_loops]):
            if value > 0:

                main_weights[0][..., idx] = main_weights[0][..., idx]/(value + 1)
                for _ in range(value):
                    main_weights[0] = np.append(main_weights[0], main_weights[0][..., idx, None], axis=-1)
                    secondary_weights[0] = np.append(secondary_weights[0], secondary_weights[0][..., [idx], :], axis=-2)

                if len(main_weights) > 1:
                    main_weights[1][idx] = main_weights[1][idx]/(value+1)
                    for _ in range(value):
                        main_weights[1] = np.append(main_weights[1], [main_weights[1][idx]], axis=0)

        self.weight_splitter.main = main_weights
        self.weight_splitter.secondary = secondary_weights

    def __modify_identity(self) -> None:
        '''
        Keeps model as is
        '''
        return

    def __modify_remove_activation(self) -> None:
        '''
        Removes the activation layer
        '''
        del self.output_config['layers'][self.config_index]

    def __modify_remove_batch_normalization(self) -> None:
        '''
        Removes a batch normalization layer
        '''
        del self.output_config['layers'][self.config_index]

        channels_to_remove = np.arange(self.weight_splitter.main[0].shape[0]).tolist()
        self.weight_splitter.main = self.__remove_bn_channels(self.weight_splitter.main, channels_to_remove)

    def __modify_remove_channels(self) -> None:
        '''
        Removes channels from a convolutional layer
        '''
        current_channels = self.output_config['layers'][self.config_index]['config']['filters']
        target_channels = self.instruction.target_units_filters_count
        self.output_config['layers'][self.config_index]['config']['filters'] = target_channels
        number_to_remove = current_channels - target_channels
        self.__remove_conv_channels(number_to_remove)

    def __modify_remove_convolution(self) -> None:
        '''
        Removes a convolutional layer
        '''
        del self.output_config['layers'][self.config_index]

        # needs to either call remove channels or add channels on itself first, and then the rest can just be cut out
        main_weights = self.weight_splitter.main
        channel_change = main_weights[0].shape[-2] - main_weights[0].shape[-1]
        if channel_change < 0:
            self.__remove_conv_channels(-channel_change)
        elif channel_change > 0:
            self.__add_conv_channels(channel_change)

        self.weight_splitter.main = []

    def __modify_remove_fc(self) -> None:
        '''
        Removes a fully connected layer
        '''
        del self.output_config['layers'][self.config_index]
        new_weights = self.weight_splitter.main[0] @ self.weight_splitter.secondary[0]
        new_bias = 0
        use_bias = False
        if len(self.weight_splitter.main) > 1:
            use_bias = True
            new_bias += self.weight_splitter.main[1] @ self.weight_splitter.secondary[0]
        if len(self.weight_splitter.secondary) > 1:
            use_bias = True
            new_bias += self.weight_splitter.secondary[1]
        self.weight_splitter.main = [new_weights]
        if use_bias:
            self.weight_splitter.main.append(new_bias)
        self.weight_splitter.secondary = []

    def __modify_remove_pooling(self) -> None:
        '''
        Removes a pooling layer
        '''
        del self.output_config['layers'][self.config_index]
        secondary_weights = self.weight_splitter.secondary
        flatten_input_shape = self.instruction.architecture_analysis.layer_analyses[self.instruction.architecture_analysis.flatten_layer].input_shape[1:]
        target_shape = flatten_input_shape + (secondary_weights[0].shape[-1],)
        secondary_weights[0] = np.reshape(secondary_weights[0], (target_shape[0]*target_shape[1],) + target_shape[2:])
        secondary_weights[0] = np.reshape(secondary_weights[0], target_shape)
        target_shape = (2*target_shape[0], 2*target_shape[1]) + target_shape[2:]
        secondary_new = np.zeros(target_shape)
        for row_idx in range(secondary_weights[0].shape[0]):
            for col_idx in range(secondary_weights[0].shape[1]):
                for i in range(2):
                    for j in range(2):
                        secondary_new[2*row_idx + i, 2*col_idx + j, : ,:] = secondary_weights[0][row_idx, col_idx, :, :]

        [row_full, col_full] = self.__is_pooling_remove_full_sized()
        if not row_full:
            secondary_new = secondary_new[:-1, ...]
        if not col_full:
            secondary_new = secondary_new[:, :-1, ...]
        secondary_new = np.reshape(secondary_new, (-1, secondary_new.shape[-1]))
        self.weight_splitter.secondary = [secondary_new]
        if len(secondary_weights) > 1:
            self.weight_splitter.secondary.append(secondary_weights[1])

    def __modify_remove_units(self) -> None:
        '''
        Removes units from a fully connected layer
        '''
        #pylint: disable=invalid-name
        u, s, vh = np.linalg.svd(self.weight_splitter.main[0], compute_uv=True, full_matrices=True)
        target_units = self.instruction.target_units_filters_count
        target_units = np.minimum(target_units, len(s))

        self.output_config['layers'][self.config_index]['config']['units'] = target_units

        s = s[:target_units]
        u = u[:, :target_units]
        vh = vh[:target_units, :target_units]
        new_weights_main = [u*s]
        if len(self.weight_splitter.main) > 1:
            new_weights_main += [vh.T @ self.weight_splitter.main[1][:target_units]]
        self.weight_splitter.main = new_weights_main
        new_weights_secondary = [vh.T @ self.weight_splitter.secondary[0][:target_units, :]]
        if len(self.weight_splitter.secondary) > 1:
            new_weights_secondary += self.weight_splitter.secondary[1:]
        self.weight_splitter.secondary = new_weights_secondary
