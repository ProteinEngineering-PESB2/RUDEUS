"""Optuna optinizer"""
import optuna
from sklearn.model_selection import cross_validate
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

class Optimizer:
    """Uses optuna to optimize a model and get best parameters"""
    def __init__(self, algorithm, X, y, metric_to_optimize, trials):
        self.algorithm = algorithm
        self.X = X
        self.y = y
        self.metrics_list = ["accuracy", "recall", "precision", "f1"]
        self.metric_to_optimize = metric_to_optimize
        self.trials = trials

    def __select_algorithm(self, algorithm, trial):
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
            model = BaggingClassifier(estimator=options_estimator,n_estimators=options_n_estimators)
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
        else:
            raise f"{algorithm} not found"
        return model

    def __objective(self, trial):
        classifier_obj = self.__select_algorithm(self.algorithm, trial)
        scores = cross_validate(classifier_obj, self.X, self.y, cv=5, scoring=self.metrics_list)
        return scores[self.metric_to_optimize].mean()

    def optimize(self):
        """Optimizes selected metric"""
        study = optuna.create_study(directions=["maximize"])
        study.optimize(self.__objective, n_trials=self.trials)
        model = eval(f"{self.algorithm}(**study.best_params)")
        return model, study.best_params