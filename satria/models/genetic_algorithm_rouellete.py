from .utils import create_new_input_features
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from .model_ensembles import EnsembleStacking
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math


class GeneticAlgorithm:
    def __init__(self,
                 X_train,
                 y_train,
                 kfold,
                 number_of_chromosome=40,
                 maximum_generations=100,
                 crossover_rate=0.78,
                 mutation_rate=0.1,
                 convergence=10
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.kfold = kfold
        self.number_of_chromosome = number_of_chromosome
        self.maximum_generations = maximum_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.convergence = convergence

        self.populations = list()
        self.populations_child_chromosomes = list()
        self.best_fitness_values = list()
        self.search_space_hyperparameter_models = self.initialize_machine_learning_model_search_spaces()

    def initialize_machine_learning_model_search_spaces(self):
        # define search space for hyperparameter as Global parameters
        search_space_hyperparameter_models = {
            'SVM': {
                'C': {
                    1: 1.1,
                    2: 1.2,
                    3: 1.3,
                    4: 1.4
                },
                'kernel': {
                    1: 'linear',
                    2: 'poly',
                    3: 'rbf',
                    4: 'sigmoid',
                },
                'gamma': {
                    1: 'scale',
                    2: 'auto',
                },
                'tol': {
                    1: 0.1,
                    2: 0.01,
                    3: 0.001,
                    4: 0.0001
                },
            },
            'DT': {
                'criterion': {
                    1: 'gini',
                    2: 'entropy',
                    3: 'log_loss',
                },
                'splitter': {
                    1: 'best',
                    2: 'random',
                },
                'max_depth': {
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 5,
                    6: None,
                },
                'min_samples_split': {
                    1: 2,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 6,
                },
                'min_samples_leaf': {
                    1: 1,
                    2: 2,
                    3: 3,
                },
            },
            'LogReg': {
                'penalty': {
                    1: 'l1',
                    2: 'l2',
                    3: 'none',
                },
                'solver': {
                    1: 'newton-cg',
                    2: 'lbfgs',
                    3: 'liblinear',
                    4: 'sag',
                    5: 'saga',
                },
                'max_iter': {
                    1: 10,
                    2: 20,
                    3: 30,
                    4: 40,
                },
                'tol': {
                    1: 0.001,
                    2: 0.0001,
                    3: 0.00001,
                },
            },
            'ANN': {
                'hidden_layer_sizes': {
                    1: (50),
                    2: (100),
                    3: (50, 50),
                    4: (50, 100),
                    5: (100, 50),
                    6: (100, 100),
                },
                'activation': {
                    1: 'identity',
                    2: 'logistic',
                    3: 'tanh',
                    4: 'relu',
                },
                'solver': {
                    1: 'lbfgs',
                    2: 'sgd',
                    3: 'adam',
                },
                'alpha': {
                    1: 0.001,
                    2: 0.0001,
                    3: 0.00001,
                },
                'batch_size': {
                    1: 32,
                    2: 64,
                    3: 128,
                },
                'learning_rate': {
                    1: 'constant',
                    2: 'invscaling',
                    3: 'adaptive',
                },
                'learning_rate_init': {
                    1: 0.001,
                    2: 0.002,
                    3: 0.003,
                    4: 0.004,
                    5: 0.005
                }
            }
        }
        return search_space_hyperparameter_models

    def initialize_populations(self):
        # inisialisasi kromosoom
        for i in range(self.number_of_chromosome):
            # define variables for each chromosome
            chromosomes = list()

            # create initial solution of each chromosome based on search space
            initial_svm_hyperparams = list()
            initial_dt_hyperparams = list()
            initial_logreg_hyperparams = list()
            initial_ann_hyperparams = list()

            # define initial fitness value and objective function
            # maximize model performance later
            initial_fitness_value = 0
            initial_objective_function = 0

            # generate random search space for each model classifiers (SVM)
            for hyperparam_name, search_spaces in self.search_space_hyperparameter_models['SVM'].items():
                random_value_param = search_spaces[random.randint(
                    1, len(search_spaces))]
                initial_svm_hyperparams.append(random_value_param)

            # generate random search space for each model classifiers (DT)
            for hyperparam_name, search_spaces in self.search_space_hyperparameter_models['DT'].items():
                random_value_param = search_spaces[random.randint(
                    1, len(search_spaces))]
                initial_dt_hyperparams.append(random_value_param)

            # generate random search space for each model classifiers (LogReg)
            for hyperparam_name, search_spaces in self.search_space_hyperparameter_models['LogReg'].items():
                random_value_param = search_spaces[random.randint(
                    1, len(search_spaces))]
                initial_logreg_hyperparams.append(random_value_param)

            # generate random search space for each model classifiers (ANN)
            for hyperparam_name, search_spaces in self.search_space_hyperparameter_models['ANN'].items():
                random_value_param = search_spaces[random.randint(
                    1, len(search_spaces))]
                initial_ann_hyperparams.append(random_value_param)

            # append initial chromosome into populations
            chromosomes.append(initial_svm_hyperparams)
            chromosomes.append(initial_dt_hyperparams)
            chromosomes.append(initial_logreg_hyperparams)
            chromosomes.append(initial_ann_hyperparams)

            # check kesesuaian nilai gene saat inisialisasi kromosom
            chromosomes = self.check_chromosomes_genes(chromosomes)

            # append solusi awal ke sekumpulan populasi
            self.populations.append(
                list((chromosomes, initial_fitness_value))
            )
        return self.populations

    def check_chromosomes_genes(self, chromosomes):
        '''
        chromosomes[0]: svm,
        chromosomes[1]: dt,
        chromosomes[2]: logreg,
        chromosomes[3]: ann
        '''
        # RULE CHECK ANN Hyperparameter Combination Rules
        # when solver ='sgd', we can use learning_rate hyperparams
        if chromosomes[3][2] != 'sgd':
            chromosomes[3][5] = 'constant'

        # RULE CHECK LOGREG Hyperparameter Combination Rules
        # check rule for solver-penalty pair
        solver_l2 = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']
        solver_none = ['newton-cg', 'lbfgs', 'sag', 'saga']
        solver_l1 = ['liblinear', 'saga']
        if chromosomes[2][0] == 'l1':
            chromosomes[2][1] = solver_l1[
                random.randint(0, len(solver_l1)-1)
            ]
        elif chromosomes[2][0] == 'none':
            chromosomes[2][1] = solver_none[
                random.randint(0, len(solver_none)-1)
            ]
        elif chromosomes[2][0] == 'l2':
            chromosomes[2][1] = solver_l2[
                random.randint(0, len(solver_l2)-1)
            ]
        else:
            # penalty is elasticnet
            chromosomes[2][1] = 'saga'

        return chromosomes

    def n_point_crossover(self, populations):
        number_of_chromosome = self.number_of_chromosome
        cr = self.crossover_rate

        # siapkan variable untuk menampung kromosom anak
        populations_child_chromosomes = list()

        # menghitung berapa kali proses crossover terjadi
        number_of_crossover = round(number_of_chromosome * cr)

        # lakukan proses crossover sebanyak number_of_crossover
        for iterate_crossover in range(number_of_crossover):
            # ambil dua kromosom induk secara acak, sehingga setiap proses crossover menghasilkan 2 kromosom anak (offspring)
            index_kromosom_induk_1 = random.randint(0, number_of_chromosome-1)
            index_kromosom_induk_2 = random.randint(0, number_of_chromosome-1)

            # antisipasi jika kedua kromomosom induk adalah sama
            while (index_kromosom_induk_1 == index_kromosom_induk_2):
                index_kromosom_induk_1 = random.randint(
                    0, number_of_chromosome-1)
                index_kromosom_induk_2 = random.randint(
                    0, number_of_chromosome-1)

            # get 2 kromosom induk by index
            kromosom_induk_1 = populations[index_kromosom_induk_1][0]
            kromosom_induk_2 = populations[index_kromosom_induk_2][0]

            # siapkan perpotongan gen untuk setiap model
            perpotongan_gen_svm = random.randint(0, len(kromosom_induk_1[0])-1)
            perpotongan_gen_dt = random.randint(0, len(kromosom_induk_1[1])-1)
            perpotongan_gen_logreg = random.randint(
                0, len(kromosom_induk_1[2])-1)
            perpotongan_gen_ann = random.randint(0, len(kromosom_induk_1[3])-1)

            # copy induk 1 menjadi anak 1 dan induk 2 menjadi anak 2
            kromosom_anak_1 = kromosom_induk_1.copy()
            kromosom_anak_2 = kromosom_induk_2.copy()

            # append gen pada induk 2 ke sisa gen anak 1
            # svm
            kromosom_anak_1[0][perpotongan_gen_svm:len(kromosom_anak_1[0])] = kromosom_induk_2[0][
                perpotongan_gen_svm:len(kromosom_induk_2[0])
            ]
            # dt
            kromosom_anak_1[1][perpotongan_gen_svm:len(kromosom_anak_1[1])] = kromosom_induk_2[1][
                perpotongan_gen_svm:len(kromosom_induk_2[1])
            ]
            # logreg
            kromosom_anak_1[2][perpotongan_gen_svm:len(kromosom_anak_1[2])] = kromosom_induk_2[2][
                perpotongan_gen_svm:len(kromosom_induk_2[2])
            ]
            # ann
            kromosom_anak_1[3][perpotongan_gen_svm:len(kromosom_anak_1[3])] = kromosom_induk_2[3][
                perpotongan_gen_svm:len(kromosom_induk_2[3])
            ]

            # append gen pada induk 1 ke sisa gen anak 2
            # svm
            kromosom_anak_2[0][perpotongan_gen_svm:len(kromosom_anak_2[0])] = kromosom_induk_1[0][
                perpotongan_gen_svm:len(kromosom_induk_1[0])
            ]
            # dt
            kromosom_anak_2[1][perpotongan_gen_svm:len(kromosom_anak_2[1])] = kromosom_induk_1[1][
                perpotongan_gen_svm:len(kromosom_induk_1[1])
            ]
            # logreg
            kromosom_anak_2[2][perpotongan_gen_svm:len(kromosom_anak_2[2])] = kromosom_induk_1[2][
                perpotongan_gen_svm:len(kromosom_induk_1[2])
            ]
            # ann
            kromosom_anak_2[3][perpotongan_gen_svm:len(kromosom_anak_2[3])] = kromosom_induk_1[3][
                perpotongan_gen_svm:len(kromosom_induk_1[3])
            ]

            # check kesesuaian hasil crossover di setiap gen
            kromosom_anak_1 = self.check_chromosomes_genes(kromosom_anak_1)
            kromosom_anak_2 = self.check_chromosomes_genes(kromosom_anak_2)

            # append kromosom anak ke populasi
            initial_fitness_value = 0
            populations_child_chromosomes.append(
                list((kromosom_anak_1, initial_fitness_value))
            )
            populations_child_chromosomes.append(
                list((kromosom_anak_2, initial_fitness_value))
            )

        self.populations_child_chromosomes = populations_child_chromosomes.copy()
        # append semua child ke chromosomes
        for child in populations_child_chromosomes:
            populations.append(child)

        return populations

    def one_point_mutation(self, populations):
        number_of_chromosome = self.number_of_chromosome
        mr = self.mutation_rate

        # menghitung berapa banyak kromosom yang akan dikenai mutasi
        number_of_chromosomes_mutated = round(number_of_chromosome*mr)

        for i in range(number_of_chromosomes_mutated):
            # get random number of chromosome will be mutated
            idx_chromosome_mutated = random.randint(0, number_of_chromosome-1)

            # get the mutated chromosomes
            mutated_chromosomes = populations[idx_chromosome_mutated][0]

            # siapkan perpotongan gen untuk setiap model untuk dimutasi (one point mutation)
            perpotongan_gen_svm = random.randint(
                0, len(mutated_chromosomes[0])-1)
            perpotongan_gen_dt = random.randint(
                0, len(mutated_chromosomes[1])-1)
            perpotongan_gen_logreg = random.randint(
                0, len(mutated_chromosomes[2])-1)
            perpotongan_gen_ann = random.randint(
                0, len(mutated_chromosomes[3])-1)

            # random insertion for genes yang terkena mutasi
            # generate random search space for each model classifiers (SVM)
            for idx, (hyperparam_name, search_spaces) in enumerate(self.search_space_hyperparameter_models['SVM'].items()):
                if perpotongan_gen_svm >= idx:
                    random_value_param = search_spaces[random.randint(
                        1, len(search_spaces))]
                    mutated_chromosomes[0][idx] = random_value_param
                    # pointer to the next element
                    perpotongan_gen_svm += 1
            # generate random search space for each model classifiers (DT)
            for idx, (hyperparam_name, search_spaces) in enumerate(self.search_space_hyperparameter_models['DT'].items()):
                if perpotongan_gen_dt >= idx:
                    random_value_param = search_spaces[random.randint(
                        1, len(search_spaces))]
                    mutated_chromosomes[1][idx] = random_value_param
                    # pointer to the next element
                    perpotongan_gen_dt += 1
            # generate random search space for each model classifiers (LogReg)
            for idx, (hyperparam_name, search_spaces) in enumerate(self.search_space_hyperparameter_models['LogReg'].items()):
                if perpotongan_gen_logreg >= idx:
                    random_value_param = search_spaces[random.randint(
                        1, len(search_spaces))]
                    mutated_chromosomes[2][idx] = random_value_param
                    # pointer to the next element
                    perpotongan_gen_logreg += 1
            # generate random search space for each model classifiers (ANN)
            for idx, (hyperparam_name, search_spaces) in enumerate(self.search_space_hyperparameter_models['ANN'].items()):
                if perpotongan_gen_ann >= idx:
                    random_value_param = search_spaces[random.randint(
                        1, len(search_spaces))]
                    mutated_chromosomes[3][idx] = random_value_param
                    # pointer to the next element
                    perpotongan_gen_ann += 1

            # check kesesuaian nilai gene saat inisialisasi kromosom
            mutated_chromosomes = self.check_chromosomes_genes(
                mutated_chromosomes)

        return populations

    def calculate_fitness(self, populations):
        '''
        Objective Function: memaksimalkan nilai accuracy
        Fitness formula: (mean akurasi dari stacking ensemble + accuracy ANN)/2
        '''
        X_train = self.X_train
        y_train = self.y_train
        kfold = self.kfold

        # train-val split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=42
        )

        # define SVM model using each chromosome solutions
        for idx_chromosome, chromosome in enumerate(populations):
            svm_hyperparams = chromosome[0][0]
            dt_hyperparams = chromosome[0][1]
            logreg_hyperparams = chromosome[0][2]
            ann_hyperparams = chromosome[0][3]

            # define ensemble stacking method
            ensemble_classifiers = EnsembleStacking(
                X_train,
                y_train,
                X_val,
                y_val,
                kfold,
                svm_params={
                    'C': svm_hyperparams[0],
                    'kernel': svm_hyperparams[1],
                    'gamma': svm_hyperparams[2],
                    'tol': svm_hyperparams[3],
                },
                dt_params={
                    'criterion': dt_hyperparams[0],
                    'splitter': dt_hyperparams[1],
                    'max_depth': dt_hyperparams[2],
                    'min_samples_split': dt_hyperparams[3],
                    'min_samples_leaf': dt_hyperparams[4],
                },
                logreg_params={
                    'penalty': logreg_hyperparams[0],
                    'solver': logreg_hyperparams[1],
                    'max_iter': logreg_hyperparams[2],
                    'tol': logreg_hyperparams[3],
                }
            ).train_ensemble()

            # create dataframe for easy understanding from ensemble classifiers stacking results
            ensemble_classifiers_results = list()
            for model in ensemble_classifiers.keys():
                ensemble_classifiers_results.append(pd.DataFrame(
                    ensemble_classifiers[model]).transpose().sort_values(by=['testing'], ascending=False))

            # concat all dataframe results
            ensemble_classifiers_results = pd.concat(
                ensemble_classifiers_results, axis=0)

            # get mean value from validation score in ensemble stacking ML classifiers
            mean_stacking_ensembles_validation_score = ensemble_classifiers_results['validation'].mean(
            )

            # create new input features before feeding in into ANN final classifier
            new_input_training_features = create_new_input_features(
                ensemble_classifiers,
                X_train,
                y_train
            )
            new_input_validation_features = create_new_input_features(
                ensemble_classifiers,
                X_val,
                y_val
            )

            # split X and y from new_input_features before feeding to ANN
            new_X_train = new_input_training_features.drop(
                ['ground_truth'], axis=1)
            new_y_train = new_input_training_features['ground_truth']
            new_X_val = new_input_validation_features.drop(
                ['ground_truth'], axis=1)
            new_y_val = new_input_validation_features['ground_truth']

            # feed new X_train and new y_train into ANN
            ann_model = MLPClassifier(
                hidden_layer_sizes=ann_hyperparams[0],
                activation=ann_hyperparams[1],
                solver=ann_hyperparams[2],
                alpha=ann_hyperparams[3],
                batch_size=ann_hyperparams[4],
                learning_rate=ann_hyperparams[5],
                learning_rate_init=ann_hyperparams[6],
            )
            ann_model.fit(new_X_train, new_y_train)
            predicted_ann_val = ann_model.predict(new_X_val)
            accuracy_ann_val = accuracy_score(y_val, predicted_ann_val)

            # calculate fitness value
            fitness_value_model = (
                mean_stacking_ensembles_validation_score + accuracy_ann_val)/2

            # replace old fitness value in each chromosome
            populations[idx_chromosome][1] = fitness_value_model

        # return populations with fitness value
        return populations

    def populations_selection(self, populations):
        """
            SELECTION POPULATION dilakukan dengan cara sorting kromosom berdasarkan nilai fitness tertinggi, pertahankan sejumlah
            number of chromsome
        """
        # (1) Hitung total fitness
        total_nilai_fitness = 0
        for idx_kromosom in range(self.number_of_chromosome):
            total_nilai_fitness += populations[idx_kromosom][1]

        # (2 & 3) Hitung Fitness relatif dan Fitness kumulatif
        fitness_relatif_list = list()
        fitness_kumulatif_list = list()
        fitness_kumulatif_prev = 0
        for idx_kromosom in range(self.number_of_chromosome):
            # hitung fitness relatif tiap kromosom
            fitness_relatif = populations[idx_kromosom][1] / \
                total_nilai_fitness
            fitness_relatif_list.append(fitness_relatif)

            # hitung fitness kumulatif tiap kromosom
            fitness_kumulatif = fitness_relatif + fitness_kumulatif_prev
            fitness_kumulatif_list.append(fitness_kumulatif)
            fitness_kumulatif_prev = fitness_kumulatif

        # (4) Pilih induk yang akan menjadi kandidat
        selected_chromosomes = list()
        for idx_kromosom in range(self.number_of_chromosome):
            # generate bilangan random [0,1]
            r = random.random()
            for idx_kromosom_selected in range(self.number_of_chromosome):
                # bandingkan nilai r dengan fitness kumulatif
                if r < fitness_kumulatif_list[idx_kromosom_selected]:
                    selected_chromosomes.append(
                        populations[idx_kromosom_selected].copy())
                    break

        return selected_chromosomes.copy()

    def plot_best_fitness_values(self, marker='*', color='b'):
        plt.figure(figsize=(12, 8))
        plt.plot(np.array(self.best_fitness_values),
                 marker=marker, color=color)
        plt.title('OPTIMAL FITNESS VALUE IN EACH GENERATIONS')
        plt.xlabel('Number of Generations')
        plt.ylabel('Best Fitness Value')
        plt.xticks(ticks=range(0, len(self.best_fitness_values)), labels=[
                   i+1 for i in range(0, len(self.best_fitness_values))])
        plt.show()

    def train(self):
        # create variable for number of convergence (stopping criteria)
        number_of_convergence = 0

        # create variable for storing best fitness value in each generation
        best_fitness = 0
        best_fitness_values = list()

        # (1) Initialize Populations
        populations = self.initialize_populations()

        for generation in range(self.maximum_generations):
            # (2) Crossover (N-point crossover)
            crossover_populations = self.n_point_crossover(populations.copy())

            # (3) Mutations (one-point mutation)
            mutated_populations = self.one_point_mutation(
                crossover_populations.copy())

            # (4) Hitung fitness function
            populations = self.calculate_fitness(mutated_populations.copy())

            # (5) Populations Selection
            new_populations = self.populations_selection(populations.copy())

            # check convergence in each generations
            if math.isclose(best_fitness, new_populations[0][1], rel_tol=1e-25):
                number_of_convergence += 1
            else:
                number_of_convergence = 0
                populations = new_populations.copy()

            # if convergence reach number_of_threshold, then break
            if number_of_convergence == self.convergence:
                break

            best_fitness = populations[0][1]
            best_fitness_values.append(best_fitness)

            print("ITERASI KE {} | BEST FITNESS = {}".format(
                generation, best_fitness))

        self.populations = populations.copy()
        self.best_fitness_values = best_fitness_values

        return populations
