from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

import numpy as np
import pandas as pd

class classification_model(object):

    def __init__(
            self,
            X_train, 
            y_train, 
            X_test, 
            y_test,
            iteration,
            name_export):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.iteration = iteration
        self.name_export = name_export

        self.scores = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
        self.keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

        self.status = None
        self.response_training = None

    #function to process average performance in cross val training process
    def process_performance_cross_val(self, performances):
        
        row_response = []
        for i in range(len(self.keys)):
            value = np.mean(performances[self.keys[i]])
            row_response.append(value)
        return row_response

    #function to obtain metrics using the testing dataset
    def get_performances(self, description, predict_label, real_label):
        accuracy_value = accuracy_score(real_label, predict_label)
        f1_score_value = f1_score(real_label, predict_label, average='weighted')
        precision_values = precision_score(real_label, predict_label, average='weighted')
        recall_values = recall_score(real_label, predict_label, average='weighted')

        row = [description, accuracy_value, f1_score_value, precision_values, recall_values]
        return row

    #function to train predictive model
    def train_predictive_model(self, name_model, clf_model):
    
        print("Train model with cross validation")
        clf_model.fit(self.X_train, self.y_train)
        response_cv = cross_validate(clf_model, self.X_train, self.y_train, cv=5, scoring=self.scores)
        performances_cv = self.process_performance_cross_val(response_cv)
        
        print("Predict responses and make evaluation")
        responses_prediction = clf_model.predict(self.X_test)
        response = self.get_performances(name_model, responses_prediction, self.y_test)
        response = response + performances_cv
        return response
    
    def make_exploration(self):
        matrix_data = []

        try:
            print("Train KNN")
            clf_model = KNeighborsClassifier()
            response_training = self.train_predictive_model("KNeighborsClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        try:
            print("Train DT")
            clf_model = DecisionTreeClassifier()
            response_training = self.train_predictive_model("DecisionTreeClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        try:
            print("Train Bagging")
            clf_model = BaggingClassifier()
            response_training = self.train_predictive_model("BaggingClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        try:
            print("Train RF")
            clf_model = RandomForestClassifier()
            response_training = self.train_predictive_model("RandomForestClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        try:
            print("Traing Extra Tree")
            clf_model = ExtraTreesClassifier()
            response_training = self.train_predictive_model("ExtraTreesClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        try:
            print("Train AdaBoostClassifier")
            clf_model = AdaBoostClassifier() 
            response_training = self.train_predictive_model("AdaBoostClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        try:
            print("Train GradientBoostingClassifier")
            clf_model = GradientBoostingClassifier() 
            response_training = self.train_predictive_model("GradientBoostingClassifier", clf_model)
            matrix_data.append(response_training)
        except:
            pass

        if len(matrix_data)>0:
            self.response_training = pd.DataFrame(matrix_data, columns=['description', 'test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'fit_time', 'score_time', 'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted', 'train_accuracy'])
            self.response_training['iteration'] = self.iteration
            self.response_training.to_csv(self.name_export, index=False)
    