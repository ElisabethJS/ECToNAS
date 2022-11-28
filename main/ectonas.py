'''
Class that performs network modifications
'''

from __future__ import annotations
from copy import deepcopy
import numpy as np
from tensorflow import keras
from util.adaptation_type import AdaptationType
from util.architecture_adapter import ArchitectureAdapter
from util.layer_analysis import LayerAnalysis
from util.layer_type import LayerType, ConvolutionalBlock
from util.runtime_parameters import DynamicParameters, StaticParameters
from util.adaptation_tools import ModificationReport, ArchitectureAnalysis, ModificationInstruction

__version__ = '3.0'
__author__ = 'EJ Schiessler'

# pylint:disable=too-few-public-methods, line-too-long

class ECToNAS():
    '''
    Class that performs network modifications
    '''
    dynamic_parameters: DynamicParameters

    def __init__(self, dynamic_parameters:DynamicParameters) -> None:
        self.dynamic_parameters = dynamic_parameters

    def generate_modified_offspring(self, model:keras.models.Sequential, parent_epoch_count:float) -> list[ModificationReport]:
        '''
        Analyses the given model and produces all potential modified models
        '''
        instructions = self.__generate_instruction_list(model, parent_epoch_count)
        architecture_adapter = ArchitectureAdapter()
        reports = []
        for instruction in instructions:
            adapted_model = architecture_adapter.Run(model, instruction)
            if adapted_model is not None:
                reports.append(ModificationReport(adapted_model, instruction.description, instruction.adaptation_type))
        return reports


    def __determine_changed_unit_filter_amount(self, layer_analysis:LayerAnalysis, adaptation_type:AdaptationType) -> int:
        '''
        Calculates the number of units or channels that should be added to or removed from the given layer
        Returns none if layer has no units or channels
        '''
        if adaptation_type not in [AdaptationType.ADD_CHANNELS, AdaptationType.REMOVE_CHANNELS,
                                   AdaptationType.ADD_UNITS, AdaptationType.REMOVE_UNITS]:
            return None

        if adaptation_type == AdaptationType.ADD_CHANNELS:
            return layer_analysis.channel_count + StaticParameters.ChannelChangeAmount
        if adaptation_type == AdaptationType.REMOVE_CHANNELS:
            return layer_analysis.channel_count - StaticParameters.ChannelChangeAmount

        unit_change_amount = np.maximum(StaticParameters.MinUnitChangeAmount, int(layer_analysis.unit_count*StaticParameters.UnitChangePercentage/100))
        if adaptation_type == AdaptationType.ADD_UNITS:
            return layer_analysis.unit_count + unit_change_amount
        if adaptation_type == AdaptationType.REMOVE_UNITS:
            return layer_analysis.unit_count - unit_change_amount

    def __generate_instruction_list(self, model:keras.models.Sequential, parent_epoch_count:float) -> list[ModificationInstruction]:
        '''
        Analyses the model and generates a list of surgery instructions
        '''
        architecture_analysis = ArchitectureAnalysis().run(model)
        instructions_list = []
        instruction_header = f' ** ep {parent_epoch_count:.2f} '
        instruction = ModificationInstruction(architecture_analysis)
        instruction.description = instruction_header + 'B'
        instruction.adaptation_type = AdaptationType.IDENTITY
        self.__determine_adaptation_steps(instruction)
        instructions_list.append(deepcopy(instruction))

        for convolutional_block in architecture_analysis.convolutional_blocks.values():
            instruction.reset()
            instruction.adaptation_type = AdaptationType.REMOVE_CONV_BLOCK
            instruction.adaptation_steps = self.__get_conv_block_removal_adaptations_list(convolutional_block)
            instruction.remove_layers = convolutional_block.involved_layers
            instruction.description = instruction_header
            instruction.description += self.__get_adaptation_description(AdaptationType.REMOVE_CONV_BLOCK).format(f'{instruction.remove_layers[0]} - {instruction.remove_layers[-1]}')
            instructions_list.append(deepcopy(instruction))

        for layer_index, layer_analysis in architecture_analysis.layer_analyses.items():
            for adaptation_type in layer_analysis.allowed_ops_at:
                instruction.reset()
                if adaptation_type == AdaptationType.REMOVE_CONV_BLOCK:
                    # already handled
                    continue
                unit_filter_changed_amount = self.__determine_changed_unit_filter_amount(layer_analysis, adaptation_type)
                if unit_filter_changed_amount is not None and unit_filter_changed_amount <= 0:
                    # cannot perform this remove action
                    continue
                instruction.target_units_filters_count = unit_filter_changed_amount
                instruction.adaptation_type = adaptation_type
                self.__determine_adaptation_steps(instruction)
                descr_formatter = []
                if adaptation_type in [AdaptationType.REMOVE_CHANNELS, adaptation_type.REMOVE_UNITS]:
                    instruction.remove_from_layer = layer_index
                    descr_formatter.append(unit_filter_changed_amount)
                elif adaptation_type in [AdaptationType.ADD_CHANNELS, adaptation_type.ADD_UNITS]:
                    instruction.add_to_layer = layer_index
                    descr_formatter.append(unit_filter_changed_amount)
                elif adaptation_type in [AdaptationType.REMOVE_FC]:
                    instruction.remove_layers = [layer_index]
                descr_formatter.append(layer_index)
                description = self.__get_adaptation_description(adaptation_type).format(*descr_formatter)
                instruction.description = instruction_header + description
                instructions_list.append(deepcopy(instruction))

            for adaptation_type in layer_analysis.allowed_ops_before:
                instruction.reset()
                instruction.adaptation_type = adaptation_type
                self.__determine_adaptation_steps(instruction)
                instruction.description = instruction_header + self.__get_adaptation_description(adaptation_type).format(layer_index)
                instruction.add_before_layer = layer_index
                instructions_list.append(deepcopy(instruction))

        return instructions_list

    def __determine_adaptation_steps(self, instruction:ModificationInstruction) -> None:
        '''
        Checks the adaptation type to see if individual steps need to be listed & updates modification instructions
        accordingly
        '''
        if instruction.adaptation_type in [AdaptationType.IDENTITY, AdaptationType.ADD_CHANNELS, AdaptationType.ADD_DIMENSION,
                                           AdaptationType.ADD_FC, AdaptationType.ADD_UNITS, AdaptationType.REMOVE_CHANNELS,
                                           AdaptationType.REMOVE_FC, AdaptationType.REMOVE_UNITS]:
            instruction.adaptation_steps = [instruction.adaptation_type]
            return
        if instruction.adaptation_type == AdaptationType.ADD_CONV_BLOCK:
            instruction.adaptation_steps = [AdaptationType.ADD_DIMENSION , AdaptationType.ADD_BATCH_NORMALIZTATION,
                                            AdaptationType.ADD_ACTIVATION]
            instruction.adaptation_steps.append(self.__get_pooling_operation())
            instruction.adaptation_steps.append(AdaptationType.ADD_CONVOLUTION)
            return

    def __get_adaptation_description(self, adaptation_type:AdaptationType) -> str:
        '''
        Returns a description string based on the passed adaptation type
        '''

        description_by_type = {
            AdaptationType.ADD_CHANNELS: '+ {} CH',
            AdaptationType.ADD_CONV_BLOCK: '+ CB',
            AdaptationType.ADD_FC: '+ FC',
            AdaptationType.ADD_UNITS: '+ {} U',
            AdaptationType.IDENTITY: 'B',
            AdaptationType.REMOVE_CHANNELS: '- {} CH',
            AdaptationType.REMOVE_CONV_BLOCK: '- CB',
            AdaptationType.REMOVE_FC: '- FC',
            AdaptationType.REMOVE_UNITS: '- {} U',
            AdaptationType.UNSPECIFIED: ''
        }

        description = description_by_type.get(adaptation_type, 'unknown type')
        description += ' L {}'
        return description

    def __get_conv_block_removal_adaptations_list(self, convolutional_block:ConvolutionalBlock) -> list[AdaptationType]:
        '''
        Determines the order in which layers of a convolutional block need to be removed
        '''
        result = []
        for layer_type in convolutional_block.involved_layer_types:
            if layer_type == LayerType.ACTIVATION:
                result.append(AdaptationType.REMOVE_ACTIVATION)
            elif layer_type == LayerType.BATCH_NORM:
                result.append(AdaptationType.REMOVE_BATCH_NORMALIZATION)
            elif layer_type in [LayerType.AVGPOOL2D, LayerType.MAXPOOL2D]:
                result.append(AdaptationType.REMOVE_POOL)
            elif layer_type == LayerType.CONV2D:
                result.append(AdaptationType.REMOVE_CONVOLUTION)
        return result

    def __get_pooling_operation(self) -> AdaptationType:
        '''
        Determine which pooling operation to use
        '''
        if StaticParameters.PoolingTypeRandom:
            if self.dynamic_parameters.rng.choice(1) == 1:
                return AdaptationType.ADD_POOL_AVG
            else:
                return AdaptationType.ADD_POOL_MAX
        elif StaticParameters.PoolingTypeUsed in ['max', 'Max', 'MaxPool', 'MaxPooling2D']:
            return AdaptationType.ADD_POOL_MAX
        else:
            return AdaptationType.ADD_POOL_AVG
