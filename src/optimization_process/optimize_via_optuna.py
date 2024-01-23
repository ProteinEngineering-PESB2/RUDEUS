import optuna
import numpy as np
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split, cross_validate
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import os
import pandas as pd
import sys
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

def prepare_dataset( test_size, X, y, random_state):
    rus = RandomUnderSampler(random_state=random_state, replacement=True)
    X, y = rus.fit_resample(X, y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def objective(trial):
    classifier_obj = select_algorithm(algorithm, trial)
    scores = cross_validate(classifier_obj, X_train, y_train, cv=5, scoring=metrics_list)
    return scores["test_recall"].mean() #puede ser recall

def select_algorithm(algorithm, trial):
    if algorithm == "SVC":
        options_kernel = trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
        options_C = trial.suggest_float("C", 0, 2)
        options_degree = trial.suggest_int("degree", 2, 10)
        options_coef0 = trial.suggest_float("coef0", 0, 5)
        model = SVC(
            C=options_C,
            kernel = options_kernel,
            degree = options_degree,
            coef0 = options_coef0,
            probability=True)
        
    elif algorithm == "KNeighborsClassifier":
        options_n_neighbors = trial.suggest_int("n_neighbors", 3, 30)
        options_leaf_size = trial.suggest_int("leaf_size", 3, 50)
        options_algorithm = trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
        options_metric = trial.suggest_categorical("metric", ["cityblock", "euclidean", "manhattan", "minkowski"])
        model = KNeighborsClassifier(
            n_neighbors=options_n_neighbors,
            leaf_size=options_leaf_size,
            algorithm=options_algorithm,
            metric=options_metric
        )
    elif algorithm == "GaussianProcessClassifier":
        options_n_restarts_optimizer = trial.suggest_int("n_restarts_optimizer", 0, 5)
        options_max_iter_predict = trial.suggest_int("max_iter_predict", 100, 1000)
        model = GaussianProcessClassifier(
            n_restarts_optimizer=options_n_restarts_optimizer,
            max_iter_predict=options_max_iter_predict
        )

    elif algorithm == "GaussianNB":
        model = GaussianNB()

    elif algorithm == "DecisionTreeClassifier":
        options_criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        options_splitter = trial.suggest_categorical("splitter", ["best", "random"])
        options_min_samples_split = trial.suggest_float("min_samples_split", 2, 50)
        options_min_samples_leaf = trial.suggest_float("min_samples_leaf", 1, 50)
        options_min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0, 0.5)
        options_max_depth = trial.suggest_int("max_depth", 1, 100)
        options_min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0, 100)
        options_max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 200)
        model = DecisionTreeClassifier(
            criterion=options_criterion,
            splitter=options_splitter,
            min_samples_split=options_min_samples_split,
            min_samples_leaf=options_min_samples_leaf,
            min_weight_fraction_leaf=options_min_weight_fraction_leaf,
            max_depth=options_max_depth,
            min_impurity_decrease=options_min_impurity_decrease,
            max_leaf_nodes=options_max_leaf_nodes
        )
    elif algorithm == "BaggingClassifier":
        options_estimator = trial.suggest_categorical("estimator", [SVC, DecisionTreeClassifier, KNeighborsClassifier])
        options_n_estimators = trial.suggest_int("n_estimators", 10, 5000)

        model = BaggingClassifier(
            estimator=options_estimator,
            n_estimators=options_n_estimators
        )

    elif algorithm == "RandomForestClassifier":
        options_n_estimators = trial.suggest_int("n_estimators", 10, 5000)
        options_criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        options_min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        options_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
        options_max_features= trial.suggest_categorical("max_features", ["sqrt", "log2"])

        model = RandomForestClassifier(
            n_estimators=options_n_estimators,
            criterion=options_criterion,
            min_samples_split=options_min_samples_split,
            min_samples_leaf=options_min_samples_leaf,
            max_features=options_max_features
        )
    
    elif algorithm == "ExtraTreesClassifier":
        options_n_estimators = trial.suggest_int("n_estimators", 10, 5000)
        options_criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        options_min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        options_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
        options_max_features= trial.suggest_categorical("max_features", ["sqrt", "log2"])

        model = ExtraTreesClassifier(
            n_estimators=options_n_estimators,
            criterion=options_criterion,
            min_samples_split=options_min_samples_split,
            min_samples_leaf=options_min_samples_leaf,
            max_features=options_max_features
        )

    elif algorithm == "AdaBoostClassifier":
        options_n_estimators = trial.suggest_int("n_estimators", 10, 5000)
        options_learning_rate = trial.suggest_float("learning_rate", 0, 0.5)
        options_algorithm = trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])
        model = AdaBoostClassifier(
            n_estimators=options_n_estimators,
            learning_rate=options_learning_rate,
            algorithm=options_algorithm
        )

    elif algorithm == "GradientBoostingClassifier":
        options_loss = trial.suggest_categorical("loss", ['log_loss', 'exponential'])
        options_learning_rate = trial.suggest_float("learning_rate", 0, 0.5)
        options_n_estimators = trial.suggest_int("n_estimators", 10, 5000)
        options_criterion = trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"])
        options_min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        options_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
        options_max_depth = trial.suggest_int("max_depth", 1, 10)
        options_max_features= trial.suggest_categorical("max_features", ["sqrt", "log2"])
        options_n_iter_no_change = trial.suggest_int("n_iter_no_change", 1, 10)
        model = GradientBoostingClassifier(
            loss=options_loss,
            learning_rate=options_learning_rate,
            n_estimators=options_n_estimators,
            criterion=options_criterion,
            min_samples_split=options_min_samples_split,
            min_samples_leaf=options_min_samples_leaf,
            max_depth=options_max_depth,
            max_features=options_max_features,
            n_iter_no_change=options_n_iter_no_change
        )
    return model

if __name__ == "__main__":

    doc_config = open(sys.argv[1], 'r')

    df_data = pd.read_csv(doc_config.readline().replace("\n", ""))
    algorithm = doc_config.readline().replace("\n", "")
    output_file = doc_config.readline().replace("\n", "")
    name_response = doc_config.readline().replace("\n", "")
    trials = 100

    doc_config.close()

    if name_response == "dna_interaction":
        df_positive = df_data[df_data[name_response] == 1]
        df_negative = df_data[df_data[name_response] == 0]
    else:
        df_positive = df_data[df_data[name_response] == "single"]
        df_positive[name_response] = 1

        df_negative = df_data[df_data[name_response] == "double"]
        df_negative[name_response] = 0
        
    df_data_negative_shuffle = shuffle(df_negative, random_state=42)
    df_data_negative_to_train = df_data_negative_shuffle[:len(df_positive)]

    df_to_train = pd.concat([df_positive, df_data_negative_to_train], axis=0)

    response = df_to_train[name_response]
    df_to_train = df_to_train.drop(columns=[name_response])

    X_train, X_test, y_train, y_test = train_test_split(df_to_train, response, random_state=42, test_size=0.3)

    metrics_list = ["accuracy", "recall", "precision", "f1"]

    study = optuna.create_study(directions=["maximize"])
    study.optimize(objective, n_trials=trials)
    model = eval(f"{algorithm}(**study.best_params)")
    model.fit(X_train, y_train)
    response_cv = cross_validate(model, X_train, y_train, cv=5,
                                     scoring=["accuracy", "precision", "f1", "recall"])
    y_pred = model.predict(X_test)
    
    cv_accuracy = np.mean(response_cv["test_accuracy"])
    cv_precision = np.mean(response_cv["test_precision"])
    cv_f1 = np.mean(response_cv["test_f1"])
    cv_recall = np.mean(response_cv["test_recall"])

    test_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    test_precision = f1_score(y_true=y_test, y_pred=y_pred)
    test_f1 = precision_score(y_true=y_test, y_pred=y_pred)
    test_recall = recall_score(y_true=y_test, y_pred=y_pred)
    test_mcc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    test_cm = confusion_matrix(y_test, y_pred).tolist()
    overfitting_ratio = test_recall / cv_recall
    columns = ["params", "train_accuracy", "train_precision", "train_f1", "train_recall",
               "test_accuracy", "test_precision", "test_f1", "test_recall", "test_mcc", "test_cm",
               "overfitting_ratio"]
    row = [[str(study.best_params), cv_accuracy, cv_precision, cv_f1, cv_recall,
        test_accuracy, test_precision, test_f1, test_recall, test_mcc, test_cm,
        overfitting_ratio]]
    df = pd.DataFrame(columns=columns, data= row)
    print(df)
    df.to_csv(output_file, index=False)
