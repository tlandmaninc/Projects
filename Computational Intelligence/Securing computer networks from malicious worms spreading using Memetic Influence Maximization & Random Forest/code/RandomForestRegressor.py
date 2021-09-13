import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import datetime
from HostNode import *
from DataGeneration import *
from IndependentCascadesModel import *
import pickle


class RandomForest:

    def __init__(self, df):
        self.__df = df

    def regressor_preprocessing(self, data):
        '''
        1) convert the str data into numbers
        2) create a vector of the labels "the y" of the data
        does NOT split to train and test
        '''
        data_labels = data[['Y']]
        data = data.drop(columns=['Y', 'node id'])

        # Convert the str data into numbers:
        obj_data = data.select_dtypes(include=['object']).copy()
        for col in obj_data:
            data[col] = data[col].astype('category').cat.codes

        return data, data_labels

    def init_ml_results(self):
        '''
        initialize a Dataframe for the RF regressor results
        :return: Dataframe initialized
        '''
        ml_results = pd.DataFrame(columns=['Algorithm', 'model index', 'MSE'])

        return ml_results

    def hyper_parameters_tuning_RandomForest(self, data, data_labels):
        '''
        Hyper parameters tuning for the RF model Using K=Folds Cros-Validation  .
        Creation of different models with different parameters, fit, predict and evaluate using MSE.
        To chose one preferable  model.

        :param data: X - the data for RF
        :param data_labels: Y - the data labels
        :return: RF_results_summary - A dataframe with all the models, their parameters and their MSE score.
        '''
        RF_results_summary = pd.DataFrame(
            columns=['model index', 'n_estimators', 'max_depth', 'criterion', 'min_samples_split', 'bootstrap', 'MSE'])
        n_estimators = [200, 500, 1000, 2000]  # number of trees in the forest
        max_depth = [10, 50, 100]  # max number of levels in each decision tree
        criterion = ['mse', 'mae']
        min_samples_split = [2, 5,10]  # min number of data points placed in a node before the node is split, used to control over-fitting
        bootstrap = [True, False]

        MSE_results = []
        model_index = 1
        for n in n_estimators:
            for max in max_depth:
                for c in criterion:
                    for min_sample in min_samples_split:
                        for b in bootstrap:
                            print('model index: ', model_index)
                            Regressor = RandomForestRegressor(random_state=1111, n_estimators=n, max_depth=max,
                                                              criterion=c, min_samples_split=min_sample, bootstrap=b)
                            kf = KFold(n_splits=5, shuffle=True, random_state=1111)
                            i = 1
                            # K_folds:
                            for train_index, test_index in kf.split(data, data_labels):
                                x_train_fold, x_test_fold = data.iloc[train_index], data.iloc[test_index]
                                y_train_fold, y_test_fold = data_labels.iloc[train_index], data_labels.iloc[test_index]
                                Regressor.fit(x_train_fold, y_train_fold.values.ravel())
                                y_prediction = Regressor.predict(x_test_fold)
                                mse = mean_squared_error(y_true=y_test_fold, y_pred=y_prediction)
                                MSE_results.append(mse)
                                i = i + 1  # fold number
                            row = pd.DataFrame([[model_index, n, max, c, min_sample, b, np.mean(MSE_results)]],
                                               columns=['model index', 'n_estimators', 'max_depth', 'criterion',
                                                        'min_samples_split', 'bootstrap', 'MSE'])
                            RF_results_summary = pd.concat([RF_results_summary, row])
                            print(RF_results_summary)
                            model_index += 1
        RF_results_summary['model index'] = RF_results_summary['model index'].astype(int)
        RF_results_summary = RF_results_summary.set_index('model index')
        print('RF_results_summary:\n', RF_results_summary)
        # RF_results_summary.to_csv(folder_path_results +'RF_results_summary.csv')

        min_mse_value = RF_results_summary['MSE'].min()
        min_mse_nodel_index = RF_results_summary['MSE'].idxmin()
        print('The minimum value of MSE is: ', min_mse_value, '. \n model index: ', min_mse_nodel_index)

        return RF_results_summary

    def create_feature_importance_plot(self, Regressor, data):
        '''
        Creates and saves a plot of feature importance based on the regressor model

        :param Regressor: The trained RF regressor model
        :param data: X - the data (not labeled)
        :return: None.
        '''
        feature_names = data.columns

        # sort the feature index by importance score in descending order:
        importances_index_desc = np.argsort(Regressor.feature_importances_)[::-1]
        feature_labels = [feature_names[i] for i in importances_index_desc]

        # plot:
        plt.figure(figsize=(20, 10))
        plt.bar(feature_labels, Regressor.feature_importances_[importances_index_desc])
        plt.xticks(feature_labels, rotation=10)
        plt.ylabel('Importance', fontsize=13)
        plt.xlabel('Features', fontsize=13)
        plt.title('Features Importance\n', fontsize=15)
        plt.savefig('features_importance_RF.png')
        # plt.show()

    def predict_and_evaluate(self, Regressor, data, data_labels):
        '''
        Prediction and Evaluation using MSE measure.

        :param Regressor: The trained RF regressor model
        :param data: X - the data used for RF
        :param data_labels: Y - the data labels  used for RF
        :return: y_prediction, MSE
        '''
        y_prediction = Regressor.predict(data)
        MSE = mean_squared_error(y_true=data_labels, y_pred=y_prediction)
        # print('y_prediction: ', y_prediction)
        # print('MSE results is: ', MSE)

        return y_prediction, MSE

    def create_RF_regressor(self, RF_results_summary, data, data_labels, model_index):
        '''
        Creates a regressor model - based on the parameters of the chosen model (chosen after the hyper parameters tuning phase).
        Fit model with the data.

        :param RF_results_summary: Dataframe with all the models, their parameters and their MSE score.
        :param data: X - the data used for the RF regressor.
        :param data_labels: Y - the labels of the data used for the RF regressor.
        :param model_index: the chosen model index for choosing the right parameters' values.
        :return: Regressor_model
        '''
        chosen_model_row = RF_results_summary[RF_results_summary['model index'] == model_index]

        n = chosen_model_row.at[model_index, 'n_estimators']
        max = chosen_model_row.at[model_index, 'max_depth']
        c = chosen_model_row.at[model_index, 'criterion']
        min_sample = chosen_model_row.at[model_index, 'min_samples_split']
        b = chosen_model_row.at[model_index, 'bootstrap']

        Regressor = RandomForestRegressor(random_state=1111, n_estimators=n, max_depth=max, criterion=c,
                                          min_samples_split=min_sample, bootstrap=b)
        Regressor_model = Regressor.fit(data, data_labels.values.ravel())

        return Regressor_model

    def conduct_ML_chosen_model(self, df, RF_results_summary, model_index):
        '''
        The pipeline of conducting the ML process for the chosen model: preprocessing, creating the model, predict and estimate, feature importance
        :param df:  the dataframe for ML
        :param RF_results_summary: dataframe with all the models, their parameters and their MSE score
        :param model_index: the chosen model index after the hyper parameters tuning phase
        :return: Regressor_model
        '''
        data, data_labels = self.regressor_preprocessing(df)
        Regressor_model = self.create_RF_regressor(RF_results_summary, data, data_labels, model_index)

        # Prediction:
        y_prediction, MSE = self.predict_and_evaluate(Regressor_model, data, data_labels)
        # Feature importance:
        self.create_feature_importance_plot(Regressor_model, data)

        return Regressor_model

    def hyper_parameters_tuning_phase(self, df):
        '''
        Preprocessing and hyper parameters tuning of Random Forest Regressor

        :param df: the dataframe for ML
        :return: RF_results_summary - dataframe with all the models, their parameters and their MSE score
        '''
        RF_start = datetime.datetime.now()
        data, data_labels = self.regressor_preprocessing(df)
        RF_results_summary = self.hyper_parameters_tuning_RandomForest(data, data_labels)
        RF_end = datetime.datetime.now()
        print("duration - Random Forest Regressor: {0}".format(str(RF_end - RF_start)))

        return RF_results_summary

    def arrange_df_for_RF(self, df_for_ML_combined):
        '''
        arrange the received dataframe to fit our problem for RF.
        Created the Y column as the (number of times a node was infected)/(the number of trials = 508).
        Drop irrelevant columns.

        :param df_for_ML_combined: dataframe created for RF phase
        :return:  dataframe created for RF phase - after arrangements
        '''
        df_for_ML_combined = df_for_ML_combined.drop(
            columns=['Total nodes infected', 'Iterations to convergence', 'seed', 'Unnamed: 0',
                     'Y'])  # the Y dropped is the old version's Y
        df_for_ML_combined['Y'] = df_for_ML_combined['times_infected'].div(508)
        df_for_ML_combined = df_for_ML_combined.drop(columns=['times_infected', 'ID'])
        df_for_ML_combined.rename(columns={"node": "Node Type"})

        return df_for_ML_combined

    def convert_df_to_avg_record(self, df_for_ML_combined, hostNode_id):
        '''
        Convert the hostNode ID into one record of the hostNode, with averaged Y column:
        1) Slices a given dataframe into the records of the hostNode based on its ID
        2) Converts the 3 records into one record with an averaged Y

        :param df_for_ML_combined: dataframe created for RF phase
        :param hostNode_id: the id of the chosen hostNode
        :return: one record of the hostNode, with averaged Y column
        '''
        hostNode_records = df_for_ML_combined[df_for_ML_combined['node id'] == hostNode_id]
        Ys = hostNode_records['Y']
        avg_Y = np.average(Ys)
        first_index = hostNode_records.index[0]
        hostNode_AVG_record = hostNode_records[hostNode_records.index == first_index]
        hostNode_AVG_record.at[first_index, 'Y'] = avg_Y

        return hostNode_AVG_record

    def predict_single_hostNode(self, df_for_ML_combined, hostNode_id, Regressor_model):
        '''
        Predict the Y of a single hostNode.
        1) Convert the hostNode ID into one record of the hostNode, with averaged Y column
        2) Pre processing of this record for RF regressor
        3) Predict the Y using the regressor model pickle

        :param df_for_ML_combined:  dataframe created for RF phase
        :param hostNode_id: the id of the chosen hostNode
        :param Regressor_model: The trained RF regressor model
        :return: y_prediction, MSE results
        '''
        hostNode_AVG_record = self.convert_df_to_avg_record(df_for_ML_combined, hostNode_id)
        data, data_label = self.regressor_preprocessing(hostNode_AVG_record)
        # Prediction:
        y_prediction, MSE = self.predict_and_evaluate(Regressor_model, data, data_label)

        return y_prediction, MSE

    def predict_HostNodes(self, list_hostNode_ids, df_for_ML_combined, Regressor_model):
        '''
        Predict the Y of 3 given hostNodes and average them.
        for each hostNode:
            1) Convert the hostNode ID into one record of the hostNode, with averaged Y column
            2) Pre processing of this record for RF regressor
            3) Predict the Y using the regressor model pickle

        :param list_hostNode_ids: list of hostNodes IDs
        :param df_for_ML_combined: Dataframe for the RF model
        :param Regressor_model: the trained Regressor model that was saved as a pickle
        :return: averaged Y prediction of the 3 hostNodes
        '''
        predictions = []
        print()
        for hostNode_id in list_hostNode_ids:
            hostNode_AVG_record = self.convert_df_to_avg_record(df_for_ML_combined, hostNode_id)
            data, data_label = self.regressor_preprocessing(hostNode_AVG_record)

            # Prediction:
            y_prediction, MSE = self.predict_and_evaluate(Regressor_model, data, data_label)
            predictions.append(y_prediction)

        # seeds_predictions = pd.DataFrame({'hostNode id':hostNodes_ids, 'y_prediction':predictions})
        avg_predictions = np.average(predictions)

        return avg_predictions
