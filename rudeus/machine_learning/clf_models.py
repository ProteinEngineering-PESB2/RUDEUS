"""Classification models"""
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

class ClfModel:
    """Classification Model training process"""
    def __init__(self, X_train, y_train, X_test, y_test, iteration = None, output_path = None):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test
        self.iteration = iteration
        self.output_path = output_path
        self.scores = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
        self.keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted',
                     'test_precision_weighted', 'test_accuracy']
        self.status = None
        self.response_training = None

    def __process_performance_cross_val(self, performances):
        return [np.mean(performances[key]) for key in self.keys]

    def __get_performances(self, description, predict_label, real_label):
        """Function to obtain metrics using the testing dataset"""
        accuracy_value = accuracy_score(real_label, predict_label)
        f1_score_value = f1_score(real_label, predict_label, average='weighted')
        precision_values = precision_score(real_label, predict_label, average='weighted')
        recall_values = recall_score(real_label, predict_label, average='weighted')
        cm_values = confusion_matrix(real_label, predict_label)
        row = [description, accuracy_value, f1_score_value, precision_values, recall_values, cm_values]
        return row

    def __train_predictive_model(self, name_model, clf_model):
        clf_model.fit(self.x_train, self.y_train)
        response_cv = cross_validate(clf_model, self.x_train, self.y_train, cv=5, scoring=self.scores)
        performances_cv = self.__process_performance_cross_val(response_cv)
        responses_prediction = clf_model.predict(self.x_test)
        response = self.__get_performances(name_model, responses_prediction, self.y_test)
        response = response + performances_cv
        return response

    def make_exploration(self):
        """Explore predictive models and store performance metrics"""
        matrix_data = []
        models = [SVC, KNeighborsClassifier,GaussianProcessClassifier,GaussianNB,
                  DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier,
                  ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier]
        for model in models:
            try:
                clf_model = model()
                model_name = type(clf_model).__name__
                response_training = self.__train_predictive_model(model_name, clf_model)
                matrix_data.append(response_training)
            except Exception as e:
                print(model, e)
        if len(matrix_data) > 0:
            self.response_training = pd.DataFrame(
                matrix_data,
                columns=['description', 'test_accuracy', 'test_f1_score',
                         'test_precision', 'test_recall', 'test_confussion_matrix',
                         'fit_time', 'score_time', 'train_f1_weighted', 'train_recall_weighted',
                         'train_precision_weighted', 'train_accuracy'])
            if self.iteration is not None:
                self.response_training['iteration'] = self.iteration
            if self.output_path is not None:
                self.response_training.to_csv(self.output_path, index=False)
        return self.response_training
    def train_model(self, model, params):
        """Train a single model"""
        if isinstance(model, str):
            clf_model = eval(f"{model}(**params)")
        else:
            clf_model = model(**params)
        model_name = type(clf_model).__name__
        response_training = self.__train_predictive_model(model_name, clf_model)
        self.response_training = pd.DataFrame(
            data = [response_training],
            columns=['description', 'test_accuracy', 'test_f1_score', 'test_precision',
                     'test_recall', 'test_confussion_matrix', 'fit_time', 'score_time',
                     'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted',
                     'train_accuracy'])
        return self.response_training
