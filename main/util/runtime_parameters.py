'''
All parameters needed to run ECToNAS
'''

from __future__ import annotations
import os
import numpy as np
from tensorflow import keras

from util.individual import Individual

__version__ = '3.0'
__author__ = 'EJ Schiessler'

class StaticParameters():
    '''
    Static parameters needed for running ECToNAS.
    These are never updated at runtime
    '''
    # Number of convolutional filters that can be added in one step
    ChannelChangeAmount = 3
    # Number of epochs in each competition step
    EpochsPerCompetitionStep = 5
    # Maximum number of competition steps until a winner was found
    MaxCompetitionSteps = 3
    # Maximum number of overall epochs that can be spent
    MaxTotalEpochs = 1000
    # Max number of parameters allowed for each individual model
    MaxParameterCountModel = 7999999
    # Max increase [%] in parameters during one modification
    MaxParameterIncreasePercent = 300
    # Max number of individuals per parent generation
    MaxSizeParentGeneration = 2
    # Min number of units that get added per add unit modification
    MinUnitChangeAmount = 2
    # Randomly choose between avg and max pooling
    PoolingTypeRandom = True
    # Type of pooling that should be used, applies only if PoolingTypeRandom is False
    PoolingTypeUsed = 'avg'
    # Number of epochs to be included in the rolling score calculation
    RollingEpochs = 3
    # Percentage by which number of units changes per add or remove unit modification
    UnitChangePercentage = 10
    # Number of epochs for warm up phase
    WarmUpEpochs = 10

class DynamicParameters():
    '''
    Class that keeps track of dynamic parameters during each run of ECToNAS
    '''

    ds_train: any
    ds_valid: any
    greediness_weight: float
    individual_id_counter: int
    individuals_by_id: dict[int, Individual]
    log: list[str]
    loss: keras.losses
    metrics: list[str]
    optimizer: keras.optimizers
    random_mode: bool
    rng: np.random.Generator
    run_dir: str
    scoring_key: str
    scoring_reversed: bool
    total_epochs_spent: int
    verbosity: int

    def __init__(self, **kwargs) -> None:
        '''
        Initialize user specified options and further parameters
        '''

        self.loss = kwargs.get('loss', keras.losses.sparse_categorical_crossentropy)
        self.optimizer = kwargs.get('optimizer', keras.optimizers.SGD())
        self.metrics = kwargs.get('metrics', ['accuracy'])
        self.ds_train = kwargs.get('ds_train', None)
        self.ds_valid = kwargs.get('ds_valid', None)
        self.reset(**kwargs)

    def reset(self, **kwargs) -> None:
        '''
        Resets the tracking objects for running ECToNAS
        '''
        self.greediness_weight = kwargs.get('greediness_weight', 1)
        self.individual_id_counter = 0
        self.individuals_by_id = {}
        self.log = []
        self.random_mode = kwargs.get('random_mode', False)
        self.rng = np.random.default_rng(seed = kwargs.get('seed', 42))
        self.scoring_key = kwargs.get('scoring', 'accuracy')
        self.scoring_reversed = (self.scoring_key == 'accuracy')
        self.total_epochs_spent = 0
        self.verbosity = kwargs.get('verbosity', 1)
        self.__create_run_folder(kwargs.get('model_dir', os.path.join('.','tmp')))

    def __create_run_folder(self, base_folder:str) -> None:
        '''
        Creates the folder into which all results from the current run are stored
        '''
        self.run_dir = base_folder
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
