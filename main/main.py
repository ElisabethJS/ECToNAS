'''
Neural architecture search using the Surgeon v3
'''

from __future__ import annotations
from tensorflow import keras
from util.adaptation_type import AdaptationType
from util.individual import Individual
from util.runtime_parameters import StaticParameters, DynamicParameters
from util.adaptation_tools import ModificationReport #pylint: disable=unused-import
from ectonas import ECToNAS

# pylint:disable=line-too-long

__version__ = '3.0'
__author__ = 'EJ Schiessler'

class NeuralArchitectureSearch():
    '''
    Neural architecture search using the Surgeon
    '''
    dynamic_parameters: DynamicParameters

    def __init__(self) -> None:
        '''
        Initializes the NeuralArchitectureSearch class with static options
        '''
        # read in fixed options from static config file
        # initialize runtime tracking object

    def run(self, initial_model:keras.models.Sequential, **kwargs) -> None:
        '''
        Runs the neural architecture search algorithm
        '''
        self.dynamic_parameters = DynamicParameters(**kwargs)
        optimized_model, optimized_individual = self.__run(initial_model)
        del_model_list = kwargs.get('cleanup', False)
        self.clean_up(del_model_list, optimized_individual.individual_id)
        return optimized_model, optimized_individual

    def clean_up(self, delete_models:bool=False, keep_winner:int=None) -> None:
        '''
        Clean up crew that trashes everything which is no longer needed after a run of the Surgeon
        '''
        if not delete_models:
            return
        for individual_id in list(self.dynamic_parameters.individuals_by_id.keys()):
            if keep_winner is not None and individual_id == keep_winner:
                continue
            self.dynamic_parameters.individuals_by_id[individual_id].kill()
            del self.dynamic_parameters.individuals_by_id[individual_id]

    def __assign_individual(self, model:keras.models.Sequential, description:str, adaptation_type:AdaptationType, parent:Individual = None) -> Individual:
        '''
        Assigns the provided model to an instance of the Individual object
        '''
        individual = Individual(self.dynamic_parameters.individual_id_counter, self.dynamic_parameters.run_dir, self.dynamic_parameters.loss,
                                self.dynamic_parameters.optimizer, self.dynamic_parameters.metrics)
        self.dynamic_parameters.individual_id_counter += 1
        individual.assign_model(model)
        individual.assign_adaptation_type(adaptation_type)
        del model
        individual.assign_parent(parent)
        individual.description += description
        self.__print(f'Added individual {individual.individual_id} with descr: {individual.description}', 2)
        if self.__is_individual_degenerate(individual):
            individual.kill()
            self.dynamic_parameters.individuals_by_id.pop(individual.individual_id, None)
            del individual
            return None
        self.dynamic_parameters.individuals_by_id[individual.individual_id] = individual
        return individual

    def __assign_to_brackets(self, individuals:list[Individual], competition_round:int=0, number_of_winners:int=1) -> list[list[Individual]]:
        '''
        Individuals are assigned to brackets.
        '''
        self.dynamic_parameters.rng.shuffle(individuals)
        brackets = []
        if len(individuals) <= number_of_winners:
            brackets = [[individual] for individual in individuals]
            return brackets
        group_count = number_of_winners*2**(StaticParameters.MaxCompetitionSteps - competition_round - 1)
        for i in range(group_count):
            selected_competitors = individuals[2*i:2*i+2] + individuals[2*group_count + i::group_count]
            if len(selected_competitors) > 0:
                brackets.append(selected_competitors)
        return brackets

    def __cull_degenerates(self, individuals:list[Individual]) -> list[Individual]:
        '''
        Removes all individuals that are degenerate
        '''
        non_degenerates = [individual for individual in individuals if not self.__is_individual_degenerate(individual)]
        return non_degenerates

    def __determine_winner_by_adaptation_type(self, competitors_by_adaptation_type:dict[AdaptationType, list[Individual]]) -> list[Individual]:
        '''
        Expects a list of competitors per adaptation type
        Performs bracket style competitions until one winner per type remains
        '''
        winners = []
        if len(competitors_by_adaptation_type.keys()) < StaticParameters.MaxSizeParentGeneration:
            number_of_winners = StaticParameters.MaxSizeParentGeneration
        else:
            number_of_winners = 1
        for competitors in competitors_by_adaptation_type.values():
            winners += self.__run_bracket_competition(competitors, select_greedy=True, number_of_winners=number_of_winners)
        return winners

    def __is_individual_degenerate(self, individual:Individual) -> bool:
        '''
        Checks if a given individual violates any consistency criteria
        '''
        # Always checked
        if individual.has_nan_weights:
            self.__print(f'Model with ID {individual.individual_id} has nan weights', 2)
            return True
        if individual.parameter_count >= StaticParameters.MaxParameterCountModel:
            self.__print(f'Model with ID {individual.individual_id} exceeds absolute parameter count: {individual.parameter_count}/{StaticParameters.MaxParameterCountModel}', 2)
            return True
        if individual.parent_id == -1:
            return False

        # Checked only if the individual has a parent
        increase = (individual.parameter_count - individual.parent_parameter_count)/individual.parent_parameter_count*100
        if increase >= StaticParameters.MaxParameterIncreasePercent:
            self.__print(f'Model with ID {individual.individual_id} exceeds parameter increase cap: {increase:.2f}/{StaticParameters.MaxParameterIncreasePercent} %', 2)
            return True
        return False

    def __perform_initialization_steps(self, initial_model:keras.models.Sequential) -> list[Individual]:
        '''
        Initializes the first individual, performs pretraining and security checks.
        Initializes the parent generation
        '''
        starting_individual = self.__assign_individual(initial_model, 'Base', AdaptationType.IDENTITY, None)
        if starting_individual is None:
            self.__print('Provided model is degenerate, Surgeon cannot run', 0)
            return None
        self.__train_individuals([starting_individual], StaticParameters.WarmUpEpochs)
        return [starting_individual]

    def __print(self, message:str, verbosity_level:int = 1) -> None:
        '''
        Prints info to console based on verbosity level
        Logs are always updated
        '''
        if verbosity_level > 1:
            message = '  ' + message
        self.dynamic_parameters.log.append(message)
        if verbosity_level <= self.dynamic_parameters.verbosity:
            print(message)

    def __produce_offspring_population(self, parent:Individual) -> list[Individual]:
        '''
        Produce all potential offspring for the given parent
        Reject degenerates and provide recuperative training to remaining children.
        '''
        surgeon = ECToNAS(self.dynamic_parameters)
        child_population = []
        parent_model = parent.load_model()
        surgery_reports = surgeon.generate_modified_offspring(parent_model, parent.epochs_trained)
        del parent_model
        for report in surgery_reports:
            child = self.__assign_individual(report.adapted_model, report.description, report.adaptation_type, parent)
            if child is None:
                continue
            child_population.append(child)

        return child_population

    def __provide_recuperative_training(self, individuals:list[Individual]) -> None:
        '''
        Each individual receives the required amount of recuperative training.
        '''
        for individual in individuals:
            self.__train_individuals([individual], individual.recuperative_amount_required)

    def __resolve_competition(self, competitors:list[Individual], select_greedy:bool = True) -> Individual:
        '''
        Select winner of the given list of competitors based on score or composite score if
        select_greedy is set to False. Winner is returned.
        '''
        if self.dynamic_parameters.random_mode:
            self.dynamic_parameters.rng.shuffle(competitors)
            return competitors[0]
        # Todo: maybe compare not with parent but before and after competition? how to implement this?
        alpha = self.dynamic_parameters.greediness_weight
        if select_greedy or alpha>=1:
            scoring_key = lambda individual: individual.score
            print_info = 'greedy'
        else:
            print_info = 'not greedy'
            score_increase = lambda individual: (individual.score - individual.previous_score)/individual.previous_score
            param_increase = lambda individual: (individual.parameter_count - individual.parent_parameter_count)/individual.parent_parameter_count
            scoring_key = lambda individual: score_increase(individual)*alpha - param_increase(individual)*(1-alpha)
        competitors.sort(key=scoring_key, reverse=self.dynamic_parameters.scoring_reversed) #Todo: will not work loss based
        self.__print(f'Competitor ranking ({print_info}):', 2)
        for comp in competitors:
            self.__print(f'({comp.individual_id}, {scoring_key(comp):.3f}, {comp.score:.3f}, {comp.previous_score:.3f}, {comp.parameter_count}, {comp.parent_parameter_count})', 2)
        return competitors[0]

    def __run(self, initial_model:keras.models.Sequential) -> tuple[keras.models.Sequential, Individual]:
        '''
        Main algorithm
        '''
        parent_generation = self.__perform_initialization_steps(initial_model)
        if parent_generation is None:
            self.__print('Initialization could not be performed. Aborting Surgeon', 0)
            return None

        while self.__termination_criteria_not_met():
            # Mutation phase
            self.__print(f'Mutation phase, {len(parent_generation)} parents', 1)
            child_generation = self.__run_mutation_phase(parent_generation)

            # Competition phase 1 - decision per adaptation type
            self.__print(f'Competition phase 1, {len(child_generation)} competitors', 1)
            child_generation = self.__run_competition_phase_1(child_generation)

            # Competition phase 2 - overall decision
            self.__print(f'Competition phase 2, {len(child_generation)} competitors', 1)
            child_generation = self.__run_competition_phase_2(child_generation)

            # Evolution phase
            self.__print('Evolution phase', 1)
            parent_generation = self.__run_evolution_phase(child_generation)

        # final selection
        self.__print('Done!', 1)
        final_winner = self.__resolve_competition(parent_generation, select_greedy=False)
        final_winner.prepare_plotting_data()
        winning_model = final_winner.load_model()
        return winning_model, final_winner

    def __run_bracket_competition(self, competitors:list[Individual], select_greedy:bool=True, number_of_winners:int=1) -> list[Individual]:
        '''
        Assigns the competitors into brackets and runs bracket style competition
        '''
        for competition_round in range(StaticParameters.MaxCompetitionSteps):
            brackets = self.__assign_to_brackets(competitors, competition_round, number_of_winners)
            remaining = []
            for bracket in brackets:
                self.__train_individuals(bracket, StaticParameters.EpochsPerCompetitionStep)
                remaining.append(self.__resolve_competition(bracket, select_greedy))
            competitors = remaining
        return competitors

    def __run_competition_phase_1(self, child_generation:list[Individual]) -> list[Individual]:
        '''
        Children compete against all competitors of the same adaptation type.
        One winner is determined per type.
        '''
        children_by_adaptation_type = self.__sort_by_adaptation_type(child_generation)
        child_generation = self.__determine_winner_by_adaptation_type(children_by_adaptation_type)
        return child_generation

    def __run_competition_phase_2(self, child_generation:list[Individual]) -> list[Individual]:
        '''
        All remaining children compete against each other in weighted competition.
        Number of winners is defined in StaticParameters.
        '''
        max_winners = StaticParameters.MaxSizeParentGeneration
        if len(child_generation) <= max_winners:
            return child_generation
        winners = self.__run_bracket_competition(child_generation, select_greedy=False, number_of_winners=max_winners)
        return winners

    def __run_evolution_phase(self, child_generation:list[Individual]) -> list[Individual]:
        '''
        Child generation becomes parent generation
        '''
        # Optional: further training, further culling
        return child_generation

    def __run_mutation_phase(self, parent_generation:list[Individual]) -> list[Individual]:
        '''
        Produces offspring from parent generation. Children receive recuperative training.
        Degenerate children are discarded.
        '''
        child_generation = []
        for parent in parent_generation:
            child_generation += self.__produce_offspring_population(parent)
        self.__provide_recuperative_training(child_generation)
        child_generation = self.__cull_degenerates(child_generation)
        return child_generation

    def __sort_by_adaptation_type(self, individuals:list[Individual]) -> dict:
        '''
        Sorts the given individuals by their adaptation type
        '''
        by_adaptation_type = {}
        for individual in individuals:
            if individual.adaptation_type not in by_adaptation_type:
                by_adaptation_type[individual.adaptation_type] = []
            by_adaptation_type[individual.adaptation_type].append(individual)
        return by_adaptation_type

    def __termination_criteria_not_met(self) -> bool:
        '''
        Check if the termination criteria for the Surgeon are met
        '''
        self.__print(f'Budget spent: {self.dynamic_parameters.total_epochs_spent:.2f} / {StaticParameters.MaxTotalEpochs}', 1)
        if self.dynamic_parameters.total_epochs_spent >= StaticParameters.MaxTotalEpochs:
            return False
        return True

    def __train_individuals(self, individuals:list[Individual], number_of_epochs:float) -> None:
        '''
        Trains all individuals in the given list
        '''
        for individual in individuals:
            individual.train(self.dynamic_parameters.ds_train, self.dynamic_parameters.ds_valid,
                             number_of_epochs, self.dynamic_parameters.scoring_key,
                             rolling_epochs=StaticParameters.RollingEpochs)
            self.dynamic_parameters.total_epochs_spent += number_of_epochs
