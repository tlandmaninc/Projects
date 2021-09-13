import operator
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import datetime as dt
from sklearn.metrics import make_scorer
from xgboost import plot_importance
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

########################################################################################################################


######################################## Design Of Experiements Configuration ##########################################
# DOE Hyperparameters
folds = 10
param_comb = 1000

# DOE Settings - K-Cross-Validation (K=10)
KF = KFold(n_splits=folds, shuffle=True, random_state=1111)

# Create Custom Scorers: AIC & BIC
AIC = make_scorer(calculateAIC, greater_is_better=False)
BIC = make_scorer(calculateBIC, greater_is_better=False)

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

# Creating an XGBoost Regressor - ****** MODIFY YOUR ALGORITHM
XGBR = XGBRegressor(objective='reg:squarederror', nthread=4)

# Parameter space grid for XGBoost Regressor - ****** MODIFY YOUR HYPERPARAMETERS
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.5, 1, 1.5, 2, 5],
    'subsample': [0, 0.3, 0.6, 0.8, 1.0],
    'colsample_bytree': [0, 0.3, 0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12, 15],
    'learning_rate': [0.0001, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1],
    'n_estimators': [10, 50, 100, 250, 400, 600, 1000]
}

# test_size=0.10-> MSE: 0.0033490747489302434-> 0.002958438624792246
# Splitting the data to Train & Test sets
# X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Process duration'],
#                                    df.loc[:, df.columns == 'Process duration'], test_size=0.1, random_state=1111)

# Splitting data to Features & Labels
X_train = df.loc[:, df.columns != 'Process duration']
y_train = df.loc[:, df.columns == 'Process duration']

# DOE using Randomized Search 10-Cross-Validation
random_search = RandomizedSearchCV(XGBR, param_distributions=params, n_iter=param_comb, scoring=metrics, refit='MSE',
                                   n_jobs=4, cv=KF.split(X_train, y_train), verbose=3, random_state=1111)
# Documents Experiment's start time
start_time = timer(None)

# Execute Experiment
RS = random_search.fit(X_train, y_train)

# Create the best XGBoost Regressor
bestXGB = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=400, n_jobs=4, nthread=4, num_parallel_tree=1,
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             subsample=0.8, tree_method='exact', validate_parameters=1,
             verbosity=None)

# Train the best achieved model
bestXGB.fit(X_train, y_train)

# plot feature importance
plot_importance(bestXGB, grid=False, title="XGBoost Feature Importance")
plt.show()

# Documents Experiment's ending time
timer(start_time)

# Print Experiment's Results
print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best MSE score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * -1)
print('\n Best hyperparameters:')
print(random_search.best_params_)

# Create Results report DF
results = pd.DataFrame(random_search.cv_results_)

# Extract Current Timestamp dd_mm_YY-H_M_S
now = dt.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

# Results Report output path - ****** MODIFY OUTPUT PATH FOR CSV REPORT
output_path = r"D:\MSc Degree\Courses\2nd Year\6. Selected Topics In Statistics\Project\Data\XGB-output-"+str(now)+\
              "_PC_"+str(param_comb)+".csv"

# Create Results report CSV
results.to_csv(output_path, index=False)


