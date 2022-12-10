# modelling for ensemble method
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class EnsembleStacking:
    def __init__(self,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 kfold,
                 svm_params=None,
                 dt_params=None,
                 logreg_params=None,
                 ann_params=None
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kfold = kfold
        self.svm_params = None
        self.dt_params = None
        self.logreg_params = None
        self.ann_params = None

        if svm_params != None:
            self.svm_params = svm_params
        if dt_params != None:
            self.dt_params = dt_params
        if logreg_params != None:
            self.logreg_params = logreg_params
        if ann_params != None:
            self.ann_params = ann_params

    def train_ensemble(self):
        ensemble_classifiers = {
            'svm': dict(),
            'logreg': dict(),
            'naive_bayes': dict(),
            'decision_tree': dict()
        }

        for idx, (train_index, val_index) in enumerate(self.kfold.split(self.X_train, self.y_train)):

            # split training set into train and val set
            X_latih, X_validasi = self.X_train[train_index], self.X_train[val_index]
            y_latih, y_validasi = self.y_train[train_index], self.y_train[val_index]

            # train 5 model of SVM
            if self.svm_params is None:
                svm = SVC()
            else:
                svm = SVC(
                    C=self.svm_params['C'],
                    kernel=self.svm_params['kernel'],
                    gamma=self.svm_params['gamma'],
                    tol=self.svm_params['tol'],
                )
            svm.fit(X_latih, y_latih)
            predicted_svm = svm.predict(X_validasi)
            tested_svm = svm.predict(self.X_test)

            ensemble_classifiers['svm']['model-'+str(idx+1)] = {
                'train': svm,
                'training': accuracy_score(y_latih, svm.predict(X_latih)),
                'validation': accuracy_score(y_validasi, predicted_svm),
                'testing': accuracy_score(self.y_test, tested_svm)
            }

            # train 5 model of Naive Bayes
            naive_bayes = MultinomialNB()
            naive_bayes.fit(X_latih, y_latih)
            predicted_naive_bayes = naive_bayes.predict(X_validasi)
            tested_naive_bayes = naive_bayes.predict(self.X_test)

            ensemble_classifiers['naive_bayes']['model-'+str(idx+1)] = {
                'train': naive_bayes,
                'training': accuracy_score(y_latih, naive_bayes.predict(X_latih)),
                'validation': accuracy_score(y_validasi, predicted_naive_bayes),
                'testing': accuracy_score(self.y_test, tested_naive_bayes)
            }

            # train 5 model of Decision Tree
            if self.dt_params is None:
                decision_tree = DecisionTreeClassifier()
            else:
                decision_tree = DecisionTreeClassifier(
                    criterion=self.dt_params['criterion'],
                    splitter=self.dt_params['splitter'],
                    max_depth=self.dt_params['max_depth'],
                    min_samples_split=self.dt_params['min_samples_split'],
                    min_samples_leaf=self.dt_params['min_samples_leaf'],
                )
            decision_tree.fit(X_latih, y_latih)
            predicted_decision_tree = decision_tree.predict(X_validasi)
            tested_decision_tree = decision_tree.predict(self.X_test)

            ensemble_classifiers['decision_tree']['model-'+str(idx+1)] = {
                'train': decision_tree,
                'training': accuracy_score(y_latih, decision_tree.predict(X_latih)),
                'validation': accuracy_score(y_validasi, predicted_decision_tree),
                'testing': accuracy_score(self.y_test, tested_decision_tree)

            }

            # train 5 model of logReg
            if self.logreg_params is None:
                log_reg = LogisticRegression()
            else:
                log_reg = LogisticRegression(
                    penalty=self.logreg_params['penalty'],
                    solver=self.logreg_params['solver'], max_iter=self.logreg_params['max_iter'], tol=self.logreg_params['tol'],
                )
            log_reg.fit(X_latih, y_latih)
            predicted_log_reg = log_reg.predict(X_validasi)
            tested_log_reg = log_reg.predict(self.X_test)

            ensemble_classifiers['logreg']['model-'+str(idx+1)] = {
                'train': log_reg,
                'training': accuracy_score(y_latih, log_reg.predict(X_latih)),
                'validation': accuracy_score(y_validasi, predicted_log_reg),
                'testing': accuracy_score(self.y_test, tested_log_reg)
            }
        return ensemble_classifiers
