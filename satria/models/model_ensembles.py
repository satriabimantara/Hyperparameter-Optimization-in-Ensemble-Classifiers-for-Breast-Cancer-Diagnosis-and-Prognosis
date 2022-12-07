# modelling for ensemble method
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class EnsembleStacking:
    def __init__(self, X_train, y_train, X_test, y_test, kfold):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kfold = kfold

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
            svm = SVC()
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
            decision_tree = DecisionTreeClassifier()
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
            log_reg = LogisticRegression(solver='newton-cg')
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
