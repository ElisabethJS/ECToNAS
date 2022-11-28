'''
Model individal that can be used in ECToNAS's network population
'''

from __future__ import annotations
import os
from copy import deepcopy
import numpy as np
from tensorflow import keras
from util.adaptation_type import AdaptationType

# pylint:disable=line-too-long

__version__ = '3.0'
__author__ = 'EJ Schiessler'

class Individual:
    '''
    Model individal that can be used in ECToNAS's network population
    '''

    adaptation_type: AdaptationType
    description: str
    epochs_trained: float
    has_nan_weights: bool
    history: dict
    individual_id: int
    log: list[str]
    loss: keras.losses
    metrics: list[str]
    model_path: str
    optimizer: keras.optimizers
    parameter_count: int
    parent_id: int
    parent_parameter_count: int
    previous_score: float
    plotting_data: np.ndarray
    plotting_keys: list[str]
    recuperative_amount_required: float
    score: float

    __Recuperative_amount_by_adaptation_type = {
        AdaptationType.IDENTITY: 0.1,
        AdaptationType.ADD_CHANNELS: 0.1,
        AdaptationType.ADD_CONV_BLOCK: 0.25,
        AdaptationType.ADD_FC: 0.1,
        AdaptationType.ADD_UNITS: 0.1,
        AdaptationType.REMOVE_CHANNELS: 1.0,
        AdaptationType.REMOVE_CONV_BLOCK: 1.0,
        AdaptationType.REMOVE_FC: 1.0,
        AdaptationType.REMOVE_UNITS: 1.0
    }

    def __init__(self, individual_id:int, base_path:str, loss:keras.losses,
                 optimizer:keras.optimizers, metrics: list[str]) -> None:
        '''
        Initialize the individual
        '''
        # set known parameters
        self.individual_id = individual_id
        self.loss = loss
        self.metrics = metrics
        self.model_path = os.path.join(base_path, f'model_{self.individual_id}.h5')
        self.optimizer = optimizer

        # initialize unknown parameters
        self.adaptation_type = AdaptationType.UNSPECIFIED
        self.description = ''
        self.epochs_trained = 0.0
        self.has_nan_weights = False
        self.history = {}
        self.log = []
        self.parameter_count = 0
        self.parent_id = -1
        self.previous_score = None
        self.parent_parameter_count = None
        self.plotting_data = None
        self.plotting_keys = None
        self.recuperative_amount_required = 0.0
        self.score = None

    def load_model(self) -> keras.models.Sequential:
        '''
        Loads the keras model from file and returns compiled model
        '''
        if not os.path.isfile(self.model_path):
            return None
        model = keras.models.load_model(self.model_path, compile=False)
        self.compile(model)
        return model

    def save(self, model:keras.models.Sequential) -> None:
        '''
        Writes the keras model to file
        '''
        model.save(self.model_path, overwrite=True, include_optimizer=False)

    def assign_adaptation_type(self, adaptation_type:AdaptationType) -> None:
        '''
        Assigns the adaptation type to the individual and calculates required
        recuperative training amount
        '''
        self.adaptation_type = adaptation_type
        self.recuperative_amount_required = self.__Recuperative_amount_by_adaptation_type.get(adaptation_type, 0.1)

    def assign_model(self, model:keras.models.Sequential) -> None:
        '''
        Assigns a keras model to the individual
        Calculates models parameters count
        '''
        self.save(model)
        self.parameter_count = np.sum([np.prod(weight.get_shape()) for weight in model.trainable_weights])
        self.__update_nan_weights_status(model)

    def assign_parent(self, parent:Individual) -> None:
        '''
        Assigns the parent of the given individual
        Parent's history is copied and parameter increase is calculated
        '''
        if parent is None:
            return
        self.parent_id = parent.individual_id
        self.parent_parameter_count = parent.parameter_count
        self.previous_score = parent.score
        self.description = parent.description
        self.history = deepcopy(parent.history)
        self.epochs_trained = parent.epochs_trained
        self.optimizer = deepcopy(parent.optimizer)

    def compile(self, model:keras.models.Sequential) -> None:
        '''
        Compiles the model using the specified options
        '''
        #optimizer = keras.optimizers.get(self.optimizer_identifier)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def kill(self) -> None:
        '''
        Deletes the stored model
        '''
        try:
            os.remove(self.model_path)
        except OSError:
            pass

    def train(self, ds_train, ds_valid, number_of_epochs:float, scoring_key:str,
              rolling_epochs:int=1) -> None:
        '''
        Trains the model for the specified number of epochs. History object and score are updated
        '''
        if int(number_of_epochs) > 0:
            self.__training_and_history_update__(ds_train, ds_valid, int(number_of_epochs))
        if not (isinstance(number_of_epochs, int) or number_of_epochs.is_integer()):
            remainder = number_of_epochs - int(number_of_epochs)
            training_batches = int(len(ds_train)*remainder)
            self.__training_and_history_update__(ds_train.take(training_batches), ds_valid, 1, remainder)
        # score is calulated rolling based on the last n epochs
        self.score = np.mean([score[1] for score in self.history[scoring_key][-rolling_epochs:]])

    def prepare_plotting_data(self) -> None:
        '''
        Turns the history object into plottable data
        '''
        plotting = []
        for key in self.history.keys():
            if len(plotting) == 0:
                plotting.append([x[0] for x in self.history[key]])
                self.plotting_keys = ['epochs']
            plotting.append([x[1] for x in self.history[key]])
            self.plotting_keys.append(key)
        plotting = np.array(plotting)
        self.plotting_data = plotting

    def __training_and_history_update__(self, ds_train, ds_valid, number_of_epochs:int, update_count:float=0.0) -> None:
        model = self.load_model()
        history = model.fit(ds_train, validation_data=ds_valid, epochs=number_of_epochs, verbose=0)
        self.__update_nan_weights_status(model)
        self.save(model)
        del model
        start_count = self.epochs_trained
        if update_count > 0:
            steps = [start_count + update_count]
            self.epochs_trained += update_count
        else:
            steps = list(np.arange(number_of_epochs) + start_count + 1)
            self.epochs_trained += number_of_epochs

        for key in history.history:
            if not key in self.history:
                self.history[key] = []
            self.history[key] += list(zip(steps, history.history[key]))

    def __update_nan_weights_status(self, model:keras.models.Sequential) -> None:
        for weight in model.get_weights():
            if np.isnan(weight).any():
                self.has_nan_weights = True
                break
