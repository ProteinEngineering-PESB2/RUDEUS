import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
import vapeplot
import numpy as np

vapeplot.set_palette('avanti')
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 18})

import warnings
warnings.filterwarnings('ignore')


class ModelSelection:
    def __init__(self, results_path):
        self.path_data = results_path
        self.list_documents = os.listdir(self.path_data)
        self.df_concat = None
        self.df_results_train = None
        self.df_results_test = None

    def merge_documents(self):
        list_df = []
        for element in self.list_documents:
            df_data = pd.read_csv(f"{self.path_data}{element}")
            name_values = element.split("_exploring")[0]
            df_data['encoder'] = name_values
            list_df.append(df_data)
        df_results = pd.concat(list_df, axis=0)

        self.df_results_train = df_results[['description', 'train_f1_weighted',
            'train_recall_weighted', 'train_precision_weighted', 'train_accuracy',
            'iteration', 'encoder']]

        self.df_results_train.columns = ["Algorithm", "F1", "Recall", "Precision", "Accuracy", "Iteration", "Encoder"]
        self.df_results_train['Stage'] = "Training"

        self.df_results_test = df_results[['description', 'test_accuracy', 'test_f1_score', 'test_precision',
            'test_recall', 'iteration', 'encoder']]

        self.df_results_test.columns = ["Algorithm", "Accuracy", "F1", "Precision", "Recall", "Iteration", "Encoder"]
        self.df_results_test['Stage'] = "Validating"

        self.df_concat = pd.concat([self.df_results_train, self.df_results_test], axis=0)
        return self.df_concat

    def plot_metrics_by_algorithm(self):
        fig, axes = plt.subplots(2,2, figsize=(12,12), sharex=True, sharey=True)
        ax1 = sns.boxplot(ax=axes[0][0], data=self.df_concat, x="F1", hue="Stage", y="Algorithm")
        ax2 = sns.boxplot(ax=axes[0][1], data=self.df_concat, x="Recall", hue="Stage", y="Algorithm")
        ax3 = sns.boxplot(ax=axes[1][0], data=self.df_concat, x="Precision", hue="Stage", y="Algorithm")
        ax4 = sns.boxplot(ax=axes[1][1], data=self.df_concat, x="Accuracy", hue="Stage", y="Algorithm")
        ax1.axvline(x = 0.84, ymin = 0, ymax = 1)
        ax2.axvline(x = 0.84, ymin = 0, ymax = 1)
        ax3.axvline(x = 0.84, ymin = 0, ymax = 1)
        ax4.axvline(x = 0.84, ymin = 0, ymax = 1)
        
    def plot_metrics_by_num_repr(self):
        fig, axes = plt.subplots(2,2, figsize=(14,14), sharex=True, sharey=True)
        ax1 = sns.boxplot(ax=axes[0][0], data=self.df_concat, x="F1", hue="Stage", y="Encoder")
        ax2 = sns.boxplot(ax=axes[0][1], data=self.df_concat, x="Recall", hue="Stage", y="Encoder")
        ax3 = sns.boxplot(ax=axes[1][0], data=self.df_concat, x="Precision", hue="Stage", y="Encoder")
        ax4 = sns.boxplot(ax=axes[1][1], data=self.df_concat, x="Accuracy", hue="Stage", y="Encoder")
        ax1.axvline(x = 0.84,ymin = 0,ymax = 1)
        ax2.axvline(x = 0.84,ymin = 0,ymax = 1)
        ax3.axvline(x = 0.84,ymin = 0,ymax = 1)
        ax4.axvline(x = 0.84,ymin = 0,ymax = 1)

    def __select_by_std(self, results_df):
        std_grouped_data = results_df[["Algorithm", "F1", "Recall", "Precision", "Accuracy", "Iteration", "Encoder"]].groupby(by=["Algorithm", "Encoder"]).std()        
        filter_std_accuracy_training = np.quantile(std_grouped_data['Accuracy'], .1)
        filter_std_precision_training = np.quantile(std_grouped_data['Precision'], .1)
        filter_std_recall_training = np.quantile(std_grouped_data['Recall'], .1)
        filter_std_f_score_training = np.quantile(std_grouped_data['F1'], .1)

        std_grouped_data["Accuracy_cat_std"] = (
            std_grouped_data["Accuracy"] <= filter_std_accuracy_training).astype(int)
        std_grouped_data["Precision_cat_std"] = (
            std_grouped_data["Precision"] <= filter_std_precision_training).astype(int)
        std_grouped_data["Recall_cat_std"] = (
            std_grouped_data["Recall"] <= filter_std_recall_training).astype(int)
        std_grouped_data["F-score_cat_std"] = (
            std_grouped_data["F1"] <= filter_std_f_score_training).astype(int)
        return std_grouped_data
    
    def __select_by_mean(self, results_df):
        mean_grouped_data = results_df[["Algorithm", "F1", "Recall", "Precision", "Accuracy", "Iteration", "Encoder"]].groupby(by=["Algorithm", "Encoder"]).mean()
        filter_mean_accuracy = np.quantile(mean_grouped_data['Accuracy'], .9)
        filter_mean_precision = np.quantile(mean_grouped_data['Precision'], .9)
        filter_mean_recall = np.quantile(mean_grouped_data['Recall'], .9)
        filter_mean_f_score = np.quantile(mean_grouped_data['F1'], .9)

        mean_grouped_data["Accuracy_cat_mean"] = (
            mean_grouped_data["Accuracy"] >= filter_mean_accuracy).astype(int)
        mean_grouped_data["Precision_cat_mean"] = (
            mean_grouped_data["Precision"] >= filter_mean_precision).astype(int)
        mean_grouped_data["Recall_cat_mean"] = (
            mean_grouped_data["Recall"] >= filter_mean_recall).astype(int)
        mean_grouped_data["F-score_cat_mean"] = (
            mean_grouped_data["F1"] >= filter_mean_f_score).astype(int)
        return mean_grouped_data
    
    def __count_votes(self, results_df, set_name, metric):
        matrix_data = []
        for index in results_df.index:
            algorithm = index[0]
            encoding = index[1]
            accuracy_value = results_df[f'Accuracy_cat_{metric}'][index]
            f_score_value = results_df[f'F-score_cat_{metric}'][index]
            precision_value = results_df[f'Precision_cat_{metric}'][index]
            recall_value = results_df[f'Recall_cat_{metric}'][index]
            row = [algorithm, encoding, accuracy_value, f_score_value, precision_value, recall_value]
            matrix_data.append(row)
        df_process = pd.DataFrame(matrix_data, columns=["Algorithm", "Encoder", f"mean_accuracy_{set_name}", f"mean_f_score_{set_name}", f"mean_precision_{set_name}", f"mean_recall_{set_name}"])
        return df_process
    
    def select(self, min_votes_number):
        std_grouped_data_training = self.__select_by_std(self.df_results_train)
        std_grouped_data_testing = self.__select_by_std(self.df_results_test)
        mean_grouped_data_training = self.__select_by_mean(self.df_results_train)
        mean_grouped_data_testing = self.__select_by_mean(self.df_results_test)

        df_process_std_training = self.__count_votes(std_grouped_data_training, "training", "std")
        df_process_std_testing = self.__count_votes(std_grouped_data_testing, "testing", "std")
        df_process_mean_training = self.__count_votes(mean_grouped_data_training, "training", "mean")
        df_process_mean_testing = self.__count_votes(mean_grouped_data_testing, "testing", "mean")

        df_merge = df_process_mean_training.merge(right=df_process_mean_testing, on=["Algorithm", "Encoder"])
        df_merge = df_merge.merge(right=df_process_std_testing, on=["Algorithm", "Encoder"])
        df_merge = df_merge.merge(right=df_process_std_training, on=["Algorithm", "Encoder"])
        df_merge['Voting'] = df_merge.sum(axis=1, numeric_only=True)
        df_merge.sort_values(by="Voting", ascending=False)
        df_merge = df_merge[df_merge['Voting']>min_votes_number]
        return df_merge