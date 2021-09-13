import pandas as pd
import os.path
import numpy as np
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import datetime as dt
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint


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
    K = y_true.shape[0]
    BIC = N * np.log(MSE) + K * np.log(N)
    return BIC

# calculate AIC for regression
def calculateAIC(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    N = y_true.shape[0]
    K = y_true.shape[0]
    AIC = N * np.log(MSE) + 2 * K
    return AIC

########################################################################################################################

######################################## Design Of Experiements Configuration ##########################################
# DOE Hyperparameters
folds = 10
param_comb = 200

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

destFolderName = 'C:/Users/gavangol/OneDrive - Intel Corporation/Desktop/STIS/Data.csv'

features = pd.read_csv(destFolderName)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Labels are the values we want to predict
labels = np.array(features['Process duration'])

# Remove the labels from the features
features= features.drop('Process duration', axis = 1)

# # Saving feature names for later use
# feature_list = list(features.columns)
# # Convert to numpy array
# features = np.array(features)
#
#
# # Get numerical feature importances
# importances = list(rf.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances
# [print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances];
#
#
# # Import matplotlib for plotting and use magic command for Jupyter Notebooks
# import matplotlib.pyplot as plt
# # Set the style
# plt.style.use('fivethirtyeight')
# # list of x locations for plotting
# x_values = list(range(len(importances)))
# # Make a bar chart
# plt.bar(x_values, importances, orientation = 'vertical')
# # Tick labels for x axis
# plt.xticks(x_values, feature_list, rotation='vertical')
# # Axis labels and title
# plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
# plt.show()


############################################################   HyperParameters   ################################################################

random_grid = {
    'bootstrap': [True,False],
    'max_depth': [60,100, 150],
    'max_features': [2, 3],
    'min_samples_leaf': [2,6,8 ],
    'min_samples_split': [6,10,14 ],
    'n_estimators': [50, 200, 500, 1000, 2000]
}


pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, search across different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 8,scoring=metrics, refit='MSE', cv = KF.split(features,labels), verbose=2, random_state=1111, n_jobs = -1)

# Fit the random search model
rf_random.fit(features, labels)

print("Best Parameters:")
print(rf_random.best_params_)

# Create Results report DF
results = pd.DataFrame(rf_random.cv_results_)

# Extract Current Timestamp dd_mm_YY-H_M_S
now = dt.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

output_path = 'C:/Users/gavangol/OneDrive - Intel Corporation/Desktop/STIS/RF-output-'+str(now)+'.csv'

# Create Results report CSV
results.to_csv(output_path, index=False)
