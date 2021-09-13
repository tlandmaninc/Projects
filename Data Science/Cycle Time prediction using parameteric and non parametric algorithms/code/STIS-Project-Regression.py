import operator
import pandas as pd
import numpy as np
from datetime import datetime

import sklearn
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import datetime as dt
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from matplotlib import pyplot
import matplotlib.pyplot as plt

################################################### Functions #########################################################

# Calculate total Execution Time
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# calculate BIC for regression
def calculateBIC(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    N = y_true.shape[0]
    K = y_true.shape[1]
    BIC = N * np.log(MSE) + K * np.log(N)
    return BIC

# calculate AIC for regression
def calculateAIC(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    N = y_true.shape[0]
    K = y_true.shape[1]
    AIC = N * np.log(MSE) + 2 * K
    return AIC

# Create Custom Scorers: AIC & BIC
AIC = make_scorer(calculateAIC, greater_is_better=False)
BIC = make_scorer(calculateBIC, greater_is_better=False)

# Define a function that compares the CV perfromance of a set of predetrmined models
def cv_comparison(models, X, y, cv=10):
    # Initiate a DataFrame for the averages and a list for all measures
    cv_metrics = pd.DataFrame()
    maes = []
    mses = []
    r2s = []
    aics = []
    bics = []

    # Loop through the models, run a CV, add the average scores to the DataFrame and the scores of
    # all CVs to the list
    for model in models:
        mae = -np.round(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv), 3)
        maes.append(mae)
        mae_avg = round(mae.mean(), 3)
        mse = -np.round(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv), 3)
        mses.append(mse)
        mse_avg = round(mse.mean(), 3)
        r2 = np.round(cross_val_score(model, X, y, scoring='r2', cv=cv), 3)
        r2s.append(r2)
        r2_avg = round(r2.mean(), 3)
        aic = np.round(cross_val_score(model, X, y, scoring=AIC, cv=cv), 3)
        aics.append(aic)
        aic_avg = round(aic.mean(), 3)
        bic = np.round(cross_val_score(model, X, y, scoring=BIC, cv=cv), 3)
        bics.append(bic)
        bic_avg = round(bic.mean(), 3)
        cv_metrics[str(model)] = [mae_avg, mse_avg, r2_avg, aic_avg, bic_avg]
    cv_metrics.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2', 'AIC', 'BIC']
    return cv_metrics, maes, mses, r2s, aics, bics

########################################################################################################################


######################################## Design Of Experiements Configuration ##########################################
# DOE Hyperparameters
folds = 10
param_comb = 13

# DOE Settings - K-Cross-Validation (K=10)
KF = KFold(n_splits=folds, shuffle=True, random_state=1111)

# Experiment Metrics
metrics = {"MAE": "neg_mean_absolute_error",
           "MSE": "neg_mean_squared_error",
           "R_Squared": "r2",
           "AIC": AIC,
           "BIC": BIC}

########################################################################################################################

# Path to data - ****** MODIFY DATA INPUT PATH
data_path = r"D:\MSc Degree\Courses\2nd Year\6. Selected Topics In Statistics\Project\Data\Final_Data.csv"

# Loading Data
df = pd.read_csv(data_path)

# One-hot-encoding for categorical columns
df = pd.get_dummies(df)

# Display first 5 rows
print(df.head(5))

# Creating an XGBoost Regressor -
LinearRegressionModel = LinearRegression()
RidgeRegressionModel = Ridge()
LassoRegressionModel = Lasso()

# Hyperparameter space grid for Linear Regression
params = {'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]}

# Splitting data to Features & Labels
X_train = df.loc[:, df.columns != 'Process duration']
y_train = df.loc[:, df.columns == 'Process duration']

# DOE using Randomized Search 10-Cross-Validation
random_search_ridge = RandomizedSearchCV(RidgeRegressionModel, param_distributions=params, n_iter=param_comb, scoring=metrics, refit='MSE',
                                   n_jobs=4, cv=KF.split(X_train, y_train), verbose=3, random_state=1111)

random_search_lasso = RandomizedSearchCV(LassoRegressionModel, param_distributions=params, n_iter=param_comb, scoring=metrics, refit='MSE',
                                   n_jobs=4, cv=KF.split(X_train, y_train), verbose=3, random_state=1111)

# Documents Experiment's start time
start_time = timer(None)

# Execute Experiment
RS_Ridge = random_search_ridge.fit(X_train, y_train)
RS_Lasso = random_search_lasso.fit(X_train, y_train)

# Documents Experiment's ending time
timer(start_time)

# Print Ridge Regression Experiment's Results
print('\n Ridge Regression results:')
print(random_search_ridge.cv_results_)
print('\n Best estimator:')
print(random_search_ridge.best_estimator_)
print('\n Best MSE score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search_ridge.best_score_ * -1)
print('\n Best hyperparameters:')
print(random_search_ridge.best_params_)

# Print Lasso Regression Experiment's Results
print('\n Lasso Regression results:')
print(random_search_lasso.cv_results_)
print('\n Best estimator:')
print(random_search_lasso.best_estimator_)
print('\n Best MSE score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search_lasso.best_score_ * -1)
print('\n Best hyperparameters:')
print(random_search_lasso.best_params_)

# # Create Results report DF
ridge_regression_results = pd.DataFrame(random_search_ridge.cv_results_)
lasso_regression_results = pd.DataFrame(random_search_lasso.cv_results_)

# Extract Current Timestamp dd_mm_YY-H_M_S
now = dt.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

# Results Report output path
ridge_regression_output_path = r"D:\MSc Degree\Courses\2nd Year\6. Selected Topics In Statistics\Project\Data\RidgeRegression-output-"+str(now)+\
              "_PC_"+str(param_comb)+".csv"

lasso_regression_output_path = r"D:\MSc Degree\Courses\2nd Year\6. Selected Topics In Statistics\Project\Data\LassoRegression-output-"+str(now)+\
              "_PC_"+str(param_comb)+".csv"

# Create Results reports CSVs
ridge_regression_results.to_csv(ridge_regression_output_path, index=True)
lasso_regression_results.to_csv(lasso_regression_output_path, index=True)

########################################################################################################################

############################################### Comparing Models #######################################################

# Create the models to be tested
linear_reg = LinearRegression()
ridge_reg = sklearn.linear_model.Ridge(alpha=0.1, random_state=1111)
lasso_reg = sklearn.linear_model.Lasso(alpha=0.01, random_state=1111)

# Put the models in a list to be used for Cross-Validation
models = [linear_reg, ridge_reg, lasso_reg]

# Run the Cross-Validation comparison with the models used in this analysis
comp, maes, mses, r2s, aics, bics = cv_comparison(models, X_train, y_train, 10)

# Create DataFrame
columns = ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold', '6th Fold', '7th Fold', '8th Fold',
           '9th Fold', '10th Fold']

metrics = ['MAE', 'MAE', 'MAE', 'MSE', 'MSE', 'MSE', 'R-Squared', 'R-Squared', 'R-Squared', 'AIC', 'AIC', 'AIC',
           'BIC', 'BIC', 'BIC']

MAE_Record = pd.DataFrame(maes, index=comp.columns, columns=columns)
MSE_Record = pd.DataFrame(mses, index=comp.columns, columns=columns)
R2_Record = pd.DataFrame(r2s, index=comp.columns, columns=columns)
AIC_Record = pd.DataFrame(aics, index=comp.columns, columns=columns)
BIC_Record = pd.DataFrame(bics, index=comp.columns, columns=columns)

# Concatenate data frames to a single data frame
final_results = pd.concat([MAE_Record, MSE_Record, R2_Record, AIC_Record, BIC_Record], keys=["MAE", "MSE", "R-Squared", "AIC", "BIC"])

# Calculate the mean and standard deviation per row
final_results['Average'] = final_results.mean(axis=1)
final_results['Stdev'] = final_results.std(axis=1)

# final_results['Average'] = np.round(r2_comp.mean(axis=1), 3)
print(final_results)

# Extract Current Timestamp dd_mm_YY-H_M_S
now = dt.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

# Results Report output path
final_results_path = r"D:\MSc Degree\Courses\2nd Year\6. Selected Topics In Statistics\Project\Data\Final_Regresssion_Results-output-"+str(now)+\
              "_PC_"+str(param_comb)+".csv"

# Create Results report CSV
final_results.to_csv(final_results_path, index=True)
