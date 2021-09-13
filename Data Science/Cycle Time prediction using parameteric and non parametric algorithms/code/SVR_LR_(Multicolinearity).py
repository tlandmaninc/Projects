import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import statistics


####################################################################################################################

######################################## Functions #################################################################
# calculate bic
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


# calculate AIC
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


# calculate VIC
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


if __name__ == "__main__":
    ####################################################################################################################

    ######################################## Design Of Experiements Configuration ######################################
    # init
    random_state = 42
    k_fold = 10
    test_size = 0.1
    total_coef = [0 for x in range(31)]
    total_MSE = list()
    total_MAE = list()
    total_r2 = list()
    BIC = list()
    AIC = list()
    mean_residuals = list()

    ####################################################################################################################

    ######################################## Data preprocess ###########################################################
    dataset = pd.read_csv(r"Data.csv")

    # preprocess and dummies
    dataset['LOT_TYPE'] = dataset['LOT_TYPE'].replace(['Non Prod'], 0)
    dataset['LOT_TYPE'] = dataset['LOT_TYPE'].replace(['PROD'], 1)
    dataset['LAST_LOT'] = dataset['LAST_LOT'].replace(['Y'], 1)
    dataset['LAST_LOT'] = dataset['LAST_LOT'].replace(['N'], 0)
    dum_df = pd.get_dummies(dataset, columns=["Lot calsification", "Job", "Machine"])
    y = dum_df['Process duration']
    x = dum_df.drop(['Process duration', 'Lot calsification_Nth Lot - MW Pre',
                     'Job_job 11', 'Machine_Machine3', 'Job_job 1', 'LOT_TYPE',
                     'Job_Non Prod job', 'PARTIALITY_SCORE', 'Lot calsification_Nth Lot'], axis=1)
    cols = x.columns

    ####################################################################################################################

    ######################################## Train - Test / 10-Cross validation ########################################
    # cross validation
    cv = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)
    for train_index, test_index in cv.split(x):
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # regressor = LinearRegression()
        regressor = SVR(kernel='rbf', C=25)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        residuals = y - y_pred
        mean_residuals.append(sum(residuals) / len(residuals))

        # for_VIF = x.iloc[:, :]
        # print(calc_vif(for_VIF))

        total_r2.append(r2_score(y_test, y_pred))
        total_MSE.append(mean_squared_error(y_test, y_pred))
        total_MAE.append(mean_absolute_error(y_test, y_pred))
        BIC.append(calculate_bic(len(y), mean_squared_error(y_test, y_pred), len(cols) - 1))
        AIC.append(calculate_aic(len(y), mean_squared_error(y_test, y_pred), len(cols) - 1))


    ####################################################################################################################

    ######################################## Calculate Resualts ########################################################
    MSE = sum(total_MSE) / k_fold
    MAE = sum(total_MAE) / k_fold
    r2 = sum(total_r2) / k_fold
    AIC_mean = sum(AIC) / k_fold
    BIC_mean = sum(BIC) / k_fold

    # print(feature)
    print("MSE -> ", MSE, statistics.stdev(total_MSE))
    print("MAE -> ", MAE, statistics.stdev(total_MAE))
    print("BIC -> ", BIC_mean, statistics.stdev(BIC))
    print("AIC -> ", AIC_mean, statistics.stdev(AIC))
    print("r2  -> ", r2, statistics.stdev(total_r2))
    print("++++++++++++++++++++++++++++++++++")


    ####################################################################################################################

    ######################################## Statistics test and plots #################################################
    # Betas - coefficients
    # tmp_list = list()
    # for coef, total in zip(regressor.coef_, total_coef):
    #     tmp_list.append(coef + total)
    # total_coef = tmp_list
    #
    #
    # coef_dic = dict()
    # for i in range(len(total_coef)):
    #     coef_dic[dum_df.columns[i]] = total_coef[i] / k_fold
    # a = {key: val for key, val in sorted(coef_dic.items(), key=lambda item: item[1], reverse=True)}
    # print(a)


    # plot residuals
    # plt.plot(y,residuals, 'o', color='darkblue')
    # plt.title("Residual Plot")
    # plt.ylabel("Residual")
    # # plt.show()

    # ANOVA
    # print(f_oneway(y, y_pred))

