import copy
from datetime import datetime, timedelta
import numpy as np
from sklearn import metrics
import tensorflow as tf
import pygad
from classification import keras_classification, makeMultiClassifierModel

class ClassificationOptimizer():
    '''
    This class defines the usage of genetic algorithm for optimizing a 
    classification algorithm.
    '''

    #F1 good enough for stopping the evolution
    #good_score = 0.5
    #Number of generations with no improvement for stopping the evolution
    saturate_steps = 4

    def __init__(self, X_train, X_test, Y_train, Y_test, 
        gens=10, pop=24, parents=10, use_adaptive=True) -> None:
        '''
        Args:
            X_train: Features for training.
            X_test: Features for testing.
            Y_train: Labels for training.
            Y_test: Labels for testing.
            gens: Number of generations.
            pop: Population size.
            parents: Number of parentes in the population.
            use_adaptive: Use adaptative learning or not.
        '''

        param_definitions = keras_classification
        self.genes = param_definitions['genes']
        self.discrete_genes = param_definitions['discrete_genes']
        self.gene_values = param_definitions['gene_values']
        self.discrete_gene_values = param_definitions['discrete_gene_values']
        self.gene_types_dict = param_definitions['gene_types']
        self.evolution_params()

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.gens = gens
        self.pop = pop
        self.parents = parents
        self.use_adaptive = use_adaptive
        self.log_frequency = 10
        self.create_ga()

    def evolution_params(self):
        """Generates GA parameters for formating the chromosomes"""
        self.num_genes = len(self.genes) + len(self.discrete_genes)
        self.gene_types = [self.gene_types_dict[x]
                      for x in self.genes]
        self.gene_types += [int for x in self.discrete_genes]
        self.gene_space = [
            {'low': self.gene_values[gene][0], 
            'high': self.gene_values[gene][1]}
            for gene in self.genes
        ]
        self.gene_space += [
            list(range(len(self.discrete_gene_values[gene])))
            for gene in self.discrete_genes
        ]
    
    def make_optimizer(name, lr):
        if name == 'AdamOptimizer':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif name =='AdadeltaOptimizer':
            return tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif name == 'RMSPropOptimizer':
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)

    def find_nan_lines(array):
        loc_tuples = list( zip(* map( list, np.where(array!=array))))
        return [x for x,y in loc_tuples]

    def create_classification(self, param_dict):
        """Creates a trained classification for the current
        algorithm using a dictionary with parameters
        for this algorithm.

        Args:
            param_dict (dict): Keys are classification parameter names
                and values are parameter values.

        Returns:
            Keras Classification: Classification 
                with parameters from param_dict
                and trained with self.X_train, self.Y_train.
        """
        optimizer = ClassificationOptimizer.make_optimizer('AdamOptimizer', 
                                                           param_dict['learning_rate'])
        print(param_dict)
        model = makeMultiClassifierModel(
            self.X_train, self.Y_train,
            param_dict['batch_size'],
            [param_dict['hidden1'], param_dict['hidden2']],
            optimizer, 
            param_dict['epochs'])

        return model
    
    def calc_score(self, model):
        """Tests a classification, giving it a score.

        Args:
            model (Keras Classification): classification to be
                tested.

        Returns:
            float: F1 weighted score of the model.
        """
        y_pred = model.predict(self.X_test)
        if len(ClassificationOptimizer.find_nan_lines(y_pred)) > 0:
            return 0.0
        else:
            try:
                roc_auc_score = metrics.roc_auc_score(self.Y_test, y_pred)
                average_precision_score = metrics.average_precision_score(self.Y_test, y_pred)
                #x = metrics.mean_absolute_percentage_error(self.Y_test, y_pred)
                return roc_auc_score
            except ValueError as err:
                print(err)
                return 0.0
        
    def fitness_func(self, genome):
        """Personalized fitness function.
            Takes a genome and calculates it's resulting
            score score

        Args:
            genome (list): list of gene values

        Returns:
            float: fitness value, a error score
        """
        param_dict = self.genome_to_params(genome)
        reg = self.create_classification(param_dict)
        score = self.calc_score(reg)
        print('score:', score)

        return score
    
    def genome_to_params(self, genome):
        """Converts a genome to a parameter dict

        Args:
            genome (list): list of gene values

        Returns:
            dict: parameter dict
        """
        param_dict = {}
        i = 0
        while i < len(self.genes):
            param_name = self.genes[i]
            param_value = genome[i]
            param_dict[param_name] = param_value
            i += 1
        while i < (len(self.genes)+len(self.discrete_genes)):
            discrete_i = i - len(self.genes)
            param_name = self.discrete_genes[discrete_i]
            param_value = genome[i]
            param_dict[param_name] = self.discrete_gene_values[param_name][param_value]

            i += 1
        return param_dict
    
    def find_best_fitness(self):
        maxes = [round(max(fit_vec), 3) for fit_vec in self.fitness_by_generation]
        max_gen = 0
        for i in range(len(maxes)):
            if maxes[i] > maxes[max_gen]:
                max_gen = i
        for fit_vec in self.fitness_by_generation:
            print(sorted(fit_vec))
        return maxes[max_gen], max_gen
    
    def create_ga(self):
        self.fitness_by_generation = []
        self.solutions_by_generation = []
        self.finish_type = 'MAX_GEN'
        self.best_fitness_gen = 0

        def on_generation(ga):
            #print('')

            self.fitness_by_generation.append(copy.deepcopy(ga.last_generation_fitness))
            self.solutions_by_generation.append(copy.deepcopy(ga.population))

            best_fitness, self.best_fitness_gen = self.find_best_fitness()
            current_gen = len(self.fitness_by_generation)-1
            
            new_time = datetime.now()
            self.evol_time = new_time-self.evol_start
            '''if self.next_log <= new_time:
                self.next_log = new_time + timedelta(minutes=self.log_frequency)'''
            min_per_gen = ((self.evol_time / (current_gen+1)).total_seconds())/60
            remaining_gens = self.gens - current_gen
            print(str(self.evol_time), 'totais. Geração atual:', current_gen,
                '. Minutos por geração:', str(min_per_gen),
                '. Geração do último melhoramento:', self.best_fitness_gen,
                '. Máximo de tempo restante:', str(remaining_gens*min_per_gen))

            '''if best_fitness <= ClassificationOptimizer.good_score:
                print(ClassificationOptimizer.good_score, " atingido, parando.")
                self.finish_type = 'REACH'
                return "stop"'''
            if current_gen-self.best_fitness_gen >= ClassificationOptimizer.saturate_steps:
                print(ClassificationOptimizer.saturate_steps, ' gerações sem melhorias, parando.')
                self.finish_type = 'SATURATE'
                return 'stop'
            
        def fitfunc(pygad_instance, g, id):
            value = self.fitness_func(g)
            print('id', id)
            return value

        mut_type, mut_perc = ('adaptive', [20, 10]) if self.use_adaptive else ('random', 'default')
        self.ga_instance = pygad.GA(num_generations=self.gens, num_parents_mating=self.parents, 
            sol_per_pop=self.pop, num_genes = self.num_genes, gene_type=self.gene_types, 
            gene_space=self.gene_space, on_generation=on_generation, fitness_func=fitfunc,
            mutation_type=mut_type, mutation_percent_genes=mut_perc)
    
    def solution_list(self):
        solutions_dict = {}
        for gen_i in range(len(self.fitness_by_generation)):
            for sol_i in range(len(self.solutions_by_generation[gen_i])):
                solution = self.solutions_by_generation[gen_i][sol_i].tolist()
                key = str(solution)
                if not key in solutions_dict:
                    solutions_dict[key] = list()
                fit = self.fitness_by_generation[gen_i][sol_i]
                solutions_dict[key].append(fit)
        solutions = [(eval(key), np.mean(fits)) 
                     for key, fits in solutions_dict.items()]
        return solutions
    
    def sort_solutions(self, bests):
        bests.sort(key = lambda tp: tp[1])
        return bests

    def run_ga(self):
        current_time = datetime.now()
        self.evol_start = current_time
        self.next_log = current_time + timedelta(minutes=self.log_frequency)
        self.ga_instance.run()
        self.evolved = self.best_fitness_gen > 0

        bests = self.sort_solutions(self.solution_list())
        print("Top Solutions:")
        for i in range(len(bests)):
            print(str(len(bests)-i), self.genome_to_params(bests[i][0]), bests[i][1])
        self.best_params, self.score = bests[-1]
        self.best_params = self.genome_to_params(self.best_params)
        self.reg = self.create_classification(self.best_params)
        self.evol_time = datetime.now()-self.evol_start