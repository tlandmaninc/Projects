##################################################
####       Install & loading packages         ####      
##################################################

# Package names
packages <- c("rcompanion","corrplot","xts", "chron","stinepack","tidyverse","reshape2","miscTools","randomForest",
              "xgboost","Metrics","caret","data.table","stringr","imputeTS", "caTools")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
# Packages loading
invisible(lapply(packages, library, character.only = TRUE))

# GLOBAL VARIABLES - MODIFY project_path TO YOUR PATH!!!!

project_path <- setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Create Data directory
data_path <- paste(project_path,"Data", sep = "/") 
if(!dir.exists(data_path))
  dir.create(data_path)
weather_data_path <- paste(data_path,"train_data.csv", sep = "/")
#Create Stations directory
ws_data_path <- paste(project_path, "Data/Stations/", sep ="/")
if(!dir.exists(ws_data_path))
  dir.create(ws_data_path)
#Create Models directory
XGB_models_path <- paste(project_path, "NEWTESTMODELS_Normalized_wsidYrAndMo/", sep ="/")
if(!dir.exists(XGB_models_path))
  dir.create(XGB_models_path)

models_optimization_path <- paste(project_path, "Optimization/", sep ="/")
if(!dir.exists(models_optimization_path))
  dir.create(models_optimization_path)


##################################################
####        Preprocessing Functions           ####      
##################################################

# 1. drop irrelevant columns
drop_irrelevant_cols <- function(df, cols_to_drop){
  df <- df[, -which(names(df) %in% cols_to_drop)] #df[,!(names(df) %in% cols_to_drop)]  
  print(paste("1.", toString(cols_to_drop), 'columns were dropped'), quote=FALSE)
  
  return(df)
}

# 2.1 drop records from training set that missing the temp value 
drop_useless_records <- function(df, cols_for_missing_values_removal){
  nrows_before = nrow(df)
  completeValuesVec <- complete.cases(df[, cols_for_missing_values_removal])
  slicedDF = df[completeValuesVec, ]
  nrows_after = nrow(slicedDF)
  print(paste(" 2.1",(nrows_before-nrows_after), 'records with N/As removed'), quote=FALSE)
  return(slicedDF)
}

# 2.2 drop records with only zeros
drop_zeros <- function(data, cols_to_check){
  data_sub <- data[, cols_to_check]
  indremoved <- which(apply(data_sub, 1, function(x) all(x == 0)) )
  data <- data[ -indremoved, ]
  print(paste(" 2.2", length(indremoved), 'records with too many zeros were removed'), quote=FALSE)
  return(data)
}

# 2.3 drop anomalous records of gbrd
drop_anomalous_records <- function(df, cols_for_missing_values_removal){
  nrows_before = nrow(df)
  df <- subset(df, df$gbrd<7000) #< 4500
  nrows_after = nrow(df)
  print(paste(" 2.3",(nrows_before-nrows_after), 'anomalous records of gbrd were removed'), quote=FALSE)
  return(df)
}


######## 2. Records to be removed ########
removingRecordsProcedure <- function(df){
  
  # Print status
  print("2. Removing records:", quote=FALSE)
  
  # 2.1 Removes records with missing values (N/As)
  cols_for_missing_values_removal <- c('dmax', 'temp', 'dewp', 'dmin', 'hmax', 'hmin', 'mdct')
  df <- drop_useless_records(df, cols_for_missing_values_removal)
  
  # 2.2 Removes records with too many zeros
  cols_to_check <- c("dewp", "dmax" ,"dmin" ,"hmdy" ,"hmax", "hmin", "wdct", "gust")
  df <- drop_zeros(df, cols_to_check)
  
  # 2.3 Removes records with anomalies - WE NEED TO PROOF THIS CLAIM
  #df <- drop_anomalous_records(df)
  
  return(df)
}

# Adds a new part of the day feature to data
createPartOfDayFeature <- function(df){
  # 0 - Morning, 1 - Afternoon, 2 - Evening, 3 - Night
  df$part_of_day<- as.factor(ifelse(df$hr>=6 & df$hr<=11, 0, 
                                    ifelse(df$hr>=12 & df$hr<=17, 1,
                                           ifelse (df$hr>=18 & df$hr<=23,2,3))))
  levels(df$part_of_day) <- c("0","1","2","3")
  return(df)
}


######## 3. Feature Engineering Procedure ########
featureEngineeringProcedure <- function(df){
  
  # 3.1 Adds a dew difference feature
  df$s_diff <- abs(df$smax - df$smin)
  
  # 3.2 Adds a dew difference feature
  df$dew_diff <- abs(df$dmax - df$dmin)
  
  # 3.3 Adds a humidity difference feature
  df$h_diff <- abs(df$hmax - df$hmin)
  
  # 3.4 Adds a day of the week feature {1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday, 7: Sunday}
  # df$day_of_week<-as.factor(strftime(as.Date(df$date, "%Y-%m-%d"), "%u"))
  
  # 3.5 Adds a part of the day feature {0: Morning(06:00-12:00), 1: Afternoon(12:00-18:00), 2: Evening(18:00-23:00), 3: Night(23:00-06:00)}
  df <- createPartOfDayFeature(df)
  
  print("3. New Features Engineered: s_diff, dew_diff, h_diff, day_of_week, part_of_day", quote=FALSE)
  
  return (df)
}

# 4. Convert columns to factors
cols_to_factors <- function(df, discrete_colums){
  df[discrete_colums] <- lapply(df[discrete_colums], factor)
  print(paste("4.", toString(discrete_colums), "columns were converted to factor"), quote=FALSE)
  return(df)
}


# 5. Missing Values Completion using Spline Interpolation
missingValuesCompletion <- function(df){
  # Convert mdct column to date POSIXct expression
  df$mdct <- as.POSIXct(df$mdct, format = "%Y-%m-%d %H:%M", tc="GMT")
  
  # df_as_ts <- as.xts(wsid_data_p, order.by=wsid_data_p$mdct)
  
  # Complete missing values using Spline Interpolation
  interpolated_df <- as.data.table(na_interpolation(df,option= 'spline'))
  
  return(as.data.frame(interpolated_df))
} #


# 6. Create normalized features
create_norm_features <- function(interpolated_df){
  
  interpolated_df$MinMaxNormGbrd <- ave(interpolated_df$gbrd, interpolated_df$wsid, interpolated_df$yr, interpolated_df$mo, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  interpolated_df$MinMaxNormHmdy <- ave(interpolated_df$hmdy, interpolated_df$wsid,interpolated_df$yr, interpolated_df$mo, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  interpolated_df$MinMaxNormDewdiff <- ave(interpolated_df$dew_diff, interpolated_df$wsid,interpolated_df$yr, interpolated_df$mo, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormstp <-ave(interpolated_df$stp, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormsmax <- ave(interpolated_df$smax, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormsmin<- ave(interpolated_df$smin, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormdewp<- ave(interpolated_df$dewp, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormdmax<- ave(interpolated_df$dmax, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormdmin<- ave(interpolated_df$dmin, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormhmax<- ave(interpolated_df$hmax, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormhmin<- ave(interpolated_df$hmin, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormwdct<- ave(interpolated_df$wdct, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormgust<- ave(interpolated_df$gust, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNorms_diff<- ave(interpolated_df$s_diff, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  #interpolated_df$MinMaxNormh_diff<- ave(interpolated_df$h_diff, interpolated_df$wsid, FUN=function(x) (x-min(x))/(max(x)-min(x)))
  
  interpolated_df[is.na(interpolated_df)] = 0
  
  #interpolated_df = subset(interpolated_df, select = -c(gbrd,hmdy,dew_diff,stp,dewp,wdct,gust,s_diff,h_diff) )
  
  
  return(interpolated_df)
}


# 7. Arranges data so the temperature variable will be placed in the last column
organize_data <- function(df){
  col_order <- c(seq(1,9),seq(11,ncol(df)),10)
  organized_df <- df[,col_order]#df[,col_order, with=FALSE]
  print("7. Data was organized successfully!", quote=FALSE)
  return(organized_df)
}

# replace NA with the column's median 
convert_NA_to_median <- function(df){
  nm <- names(df)[colSums(is.na(df)) != 0]
  result <- df %>% mutate_at(vars(starts_with(nm)), funs(ifelse(is.na(.),median(., na.rm = TRUE),.)))
  return(result)
}


save_columns_medians <- function(wsid,data_table){
  
  new_row <- c()
  for (col in colnames(data_table)){
    if (is.numeric(data_table[[col]][1])){
      col_median <- median(data_table[[col]])
    }
    else{
      col_median <- 0
    }
    
    new_row <- c(new_row,col_median)
  }
  
  new_df = as.data.frame(t(new_row))
  colnames(new_df) <- names(data_table)
  # medians_dataframe <- rbind(medians_dataframe, new_df);
  return(new_df)
}

convert_NA_to_median_test_set <- function(medians_dataframe,data_table){
  for (row in nrow(data_table)){
    for(col in ncol(data_table)){
      
    }
  }
}


##################################################
####    Train Data Preprocessing Procedure    ####      
##################################################

preprocessingTrainData <- function(df){
  
  ######## 1. Drops irrelevant columns ########
  cols_to_drop <- c('wsnm','elvt','lat','lon','inme','city','prov','wdsp', 
                    'prcp')
  df_removed <- drop_irrelevant_cols(df, cols_to_drop)
  
  ######## 2. Records to be removed ########
  df_removed <- removingRecordsProcedure(df_removed)
  
  ######## 3. Feature Engineering ########
  df_removed <- featureEngineeringProcedure(df_removed) 
  
  ######## 4. Factoring Columns ########
  discrete_colums <- c('yr', 'mo', 'da', 'hr')
  df_removed <- cols_to_factors(df_removed, discrete_colums)
  
  ######## 5. Missing Values Completion using Spline Interpolation ########
  print("5. Missing values completion using Spline Interpolation was performed on gbrd & gust columns", quote=FALSE)
  
  unique_wsid = sort(unique(df_removed$wsid))
  
  for(wsid in unique_wsid){
    completedValuesDF <- as.data.frame(missingValuesCompletion(df_removed[which(df_removed$wsid==wsid),]))
    #print(paste("In CompVal, type:",class(completedValuesDF),"records:",nrow(completedValuesDF),"Columns:",ncol(completedValuesDF)))
    #print(paste("In df_removed filtered, type:",class(completedValuesDF),"records:",nrow(df_removed[which(df_removed$wsid==wsid),]),"Columns:",ncol(df_removed[which(df_removed$wsid==wsid),])))
    
    df_removed[which(df_removed$wsid==wsid),] <- completedValuesDF
    
  }
  
  df_removed <- create_norm_features(df_removed)
  ######## 6. Drops columns that were required for feature Engineering phase ########
  df_interpolated<- subset(df_removed,select=-c(mdct,date))
  
  print("6. mdct & date columns were dropped", quote=FALSE)
  
  ######## 7. Organize columns - move temp to the end of data table  ########
  current_weather_station_final <- organize_data(df_interpolated)
  print(paste("After organize_data:",ncol(current_weather_station_final)))
  return(current_weather_station_final)
}

##################################################
####    Splitting data by weather stations    ####      
##################################################

# Split the data by weather stations (wsid)
split_to_wstations<- function(data, path){
  # get unique wsid
  unique_wsid <- sort(unique(data$wsid))
  for (id in unique_wsid) {
    wsid_data <- data[which(data$wsid==id),]
    wsid_data_path <- paste(path, toString(id), ".csv", sep="")
    print(paste("# Create Weather Station", id, "CSV File. Total Records:", nrow(wsid_data)), quote=FALSE)
    write.csv(wsid_data, wsid_data_path, row.names = FALSE)
  }
}
# split_to_wstations(weather_data, ws_data_path)


# Train an XGBoost tree for a given station's ID
XGBTreeTrainingProcedure <- function(df, wsid, best_RMSE){
      
  print(paste("Best RMSE of",toString(best_RMSE)))
  
  # Splitting the dataset into Training and Validation sets
  set.seed(42)
  round_number = 25000
  # split <- sample.split(df$temp, SplitRatio = 1)
  training_set <- df[,-c(1)] #subset(df[,-c(1)], split == TRUE)
  validation_set <- df[,-c(1)] #subset(df[,-c(1)], split == FALSE)
  
  # init for XGBoost
  x_train <- model.matrix(~.+0,data=subset(training_set,select=-c(temp)))
  y_train <- as.numeric(training_set$temp)
  x_val <- model.matrix(~.+0,data = subset(training_set,select=-c(temp))) # Same as x_train
  y_val <- as.numeric(training_set$temp) # Same as y_train
  print(length(x_val))
  print(length(y_val))
  print(length(x_train))
  print(length(y_train))
  # Create XGB Matrix
  #if(id %in% c(178,303,304,305,306,308,309,310,311)){
  #return (best_RMSE)
  #  }
  
  
  dtrain <- xgb.DMatrix(data = x_train, label=y_train) 
  
  # Hyperparameters configuration
  params <- list(booster = "gbtree",eval_metric='rmse', max_depth=8, verbosity=0, objective='reg:squarederror')
  
  # XGBoost Regressor
  xgb = xgboost( params = params, data = dtrain, nrounds = round_number, nfold = 10, early_stopping_rounds = 20, maximize = F)
  
  # Predict using validation set 
  y_pred <- predict(xgb, x_val)
  
  # Calculate Root Mean Squared Error (RMSE)
  RMSE <- sqrt(mse(y_val , y_pred))
  
  # Set working directory to Models folder
  setwd(XGB_models_path)
  
  # Store station's XGB model
  saveRDS(xgb, file = paste("Station_",toString(wsid),"_RMSE_",toString(RMSE),".Rds", sep = ""))
  
  # Store the best classifier
  if(RMSE < best_RMSE) {
    print(best_RMSE)
    assign("best_RMSE", RMSE)
    best_RMSE <<- RMSE
    print(best_RMSE)
    saveRDS(xgb, file = paste("Best_Model_Station_",toString(wsid),"_RMSE_",toString(best_RMSE),".Rds",sep = ""))
  }
  
  return(best_RMSE)
}


##################################################
####   Training an XGB per weather station    ####      
##################################################

trainEnsembleOfXGBs <- function(df, best_RMSE=10){
  print("IN trainEnsembleOfXGBs")
  
  # Create a vector of unique weather stations' IDs
  unique_wsid <- sort(unique(df$wsid))
  
  # Print status
  print(paste("Train an Ensemble of",toString(length(unique_wsid)),"XGBoost Trees per each weather station"))
  
  # Iterate weather stations
  for (id in unique_wsid){
    
    # Sets working directory to stations directory
    setwd(ws_data_path)
    
    # read preprocessed data of current weather station
    # current_weather_station <- read.csv(paste(toString(id),".csv", sep = ""), header=TRUE)
    current_weather_station <- subset(df, df$wsid==id)
    
    # Print Status
    print(paste("##### Station",toString(id),"was loaded succesfully with",nrow(current_weather_station), "records #####"), quote=FALSE)  
    
    # Train an XGBoost tree for current station
    best_RMSE <- XGBTreeTrainingProcedure(current_weather_station, id, best_RMSE)
    
    # Store processed data as CSV
    # write.csv(processed_weather_station, paste(toString(id),"_processed.csv", sep = ""), row.names = FALSE)
    print("###########################################################################################",quote=FALSE)
    }
  
}


# Executes Train Pipeline 
executeTrainDataPipeline <- function(){
  
  ##################################################
  ####              Loading Data                ####      
  ##################################################
  
  print("Loading Data....")
  
  # load data into a data frame object
  weather_data <- as.data.frame(fread(weather_data_path))
  
  print("Data was loaded successfully!")
  
  ##################################################
  ####          Data Preprocessing              ####      
  ##################################################
  
  print("Preprocessing Data....")
  
  # Preprocess whole data
  weather_data_processed <- preprocessingTrainData(weather_data)
  print(paste("After preprocessing:",ncol(weather_data_processed)))
  
  # Print Summary
  print(paste("# Weather data was preprocessed succesfully and ended up with",nrow(weather_data_processed), "records."), quote=FALSE)  
  
  # Create vector of unique weather stations' IDs
  unique_wsid <- sort(unique(weather_data_processed$wsid))
  
  # Remove hr & dmin (needed for Feature Engineering)
  correlated_features_to_be_removed <- c("hr")
  weather_data_final <- drop_irrelevant_cols(weather_data_processed, correlated_features_to_be_removed)
  
  
  fwrite(weather_data_final, "TrainFinalBeforeXGBTraining_WithNormsOrigRetained_RemoveSome.csv")
  
  ##################################################
  ####           Training Procedure             ####      
  ##################################################
  
  trainEnsembleOfXGBs(weather_data_final)
  
  return(weather_data_final)
}

##################################################
####        Executes Train Pipeline           ####      
##################################################

weather_data_final <- executeTrainDataPipeline()
#testload <- read.csv(file = 'TrainFinalBeforeXGBTraining.csv')
#head(carSpeeds)
#getwd()
#setwd()

#new_DF <- testload[rowSums(is.na(testload)) > 0,]


##################################################
####       Explanatory Data Analysis          ####      
##################################################

# Convert data to data frame
weather_processed_df <- as.data.frame(weather_data_processed)

# Extract Continuous features
continuos_data <- weather_processed_df[,sapply(weather_processed_df, is.numeric)]
continuos_data <- drop_useless_records(continuos_data, seq(1,ncol(continuos_data)))

# Calculate Pearson Correlation matrix
pearson_corr_matrix <- cor(continuos_data, method = "pearson")

# Visualize Pearson Correlation for Continuous Data
corrplot(pearson_corr_matrix, method ="number")

# Extract Nominal features
nominal_data <- weather_processed_df[,sapply(weather_processed_df, is.factor)]

# CramerV Correlation between Hour (hr) Vs. part_of_day - CramerV of 100%
cramerV(nominal_data$hr,nominal_data$part_of_day)



# Anomalies Testing - gbrd
# Before Preprocessing
gbrd_bp_before <- boxplot(weather_data$gbrd, main="gbrd - Before Preprocessing", 
                          ylab="gbrd")
text(x = col(gbrd_bp_before$stats) + .3, y = gbrd_bp_before$stats, 
     labels = gbrd_bp_before$stats)

# After Preprocessing
gbrd_bp_after <- boxplot(weather_processed_df$gbrd, main="gbrd Box Plot - After Preprocessing",
                         ylab="gbrd")
text(x = col(gbrd_bp_after$stats) + .3, y = gbrd_bp_after$stats, 
     labels = gbrd_bp_after$stats)


# Boxplot of MPG by Car Cylinders
boxplot(gbrd,data=weather_data_final, main="gbrd Box Plot")

# Random Forest
library("randomForest")
fit <- randomForest(weather_data_final$temp~., data=weather_data_final)

fit <- randomForest(factor(Y)~., data=df)

VI_F <- importance(fit)

library("caret")
varImp(fit)
varImpPlot(fit, type=2)
barplot(t(VI_F/sum(VI_F)))


##################################################
##################################################
####        Hyper-parameters Tuning           ####      
##################################################
##################################################


XGB_randomSearch_single_model <- function(wsid_data,
                                          nrounds,
                                          n_max_depth,
                                          n_eta,
                                          n_gamma,
                                          n_subsample){
  set.seed(42)
  
  # init for XGBoost
  data_x <- model.matrix(~.+0,data=subset(wsid_data,select=-c(wsid,temp)))
  data_y <- as.numeric(wsid_data$temp)
  
  # init watch list
  train_watch_list <- xgb.DMatrix(data = data_x,label = data_y) 
  
  
  # note to start nrounds from 200, as smaller learning rates result in errors so
  # big with lower starting points that they'll mess the scales
  tune_grid <- expand.grid(
    nrounds = nrounds,
    max_depth = sample(5:15, n_max_depth),
    eta = runif(n_eta, .01, .3),
    gamma = runif(n_gamma, 0.0, 0.7), 
    subsample = runif(n_subsample, .6, .9),
    colsample_bytree = runif(1, .5, .8), # single value (not tuned)
    min_child_weight = sample(1:40, 1) # single value (not tuned)
  )
  
  tune_control <- caret::trainControl(
    method = "cv", # cross-validation
    number = 10, # with n folds 
    verboseIter = F, # no training log
    allowParallel = FALSE, # FALSE for reproducible results 
    search = "grid"
  )
  
  xgb_tune <- caret::train(
    x = data_x,
    y = data_y,
    trControl = tune_control,
    tuneGrid = tune_grid,
    method = "xgbTree",
    booster = "gbtree",
    verbose = F,
    metric = 'RMSE',
    maximize = F,
    eval_metric = "rmse",
    early_stopping_rounds = 20,
    watchlist = list(train=train_watch_list),
    objective = "reg:squarederror"
  )
  
  return(xgb_tune)
}


XGB_randomSearch <- function(all_data, 
                             path_for_csv,
                             save_models,
                             path_for_models='',
                             nrounds, 
                             n_max_depth,
                             n_eta,
                             n_gamma,
                             n_subsample){
  
  # init the final confs dataframe
  final_confs_df <- data.frame(matrix(ncol = 9, nrow = 0))
  cols_names <- c('wsid',
                  'eta',
                  'max_depth',
                  'gamma',
                  'colsample_bytree',
                  'min_child_weight',
                  'subsample',
                  'nrounds',
                  'RMSE')
  
  colnames(final_confs_df) <- cols_names
  
  for (wsid_num in unique(all_data$wsid)){
    print(paste("Optimizing station", wsid_num,"....."))
    curr_data <- subset(all_data, all_data$wsid==wsid_num)
    
    curr_xgb_tune <- XGB_randomSearch_single_model(curr_data,
                                                   nrounds,
                                                   n_max_depth,
                                                   n_eta,
                                                   n_gamma,
                                                   n_subsample)
    
    # extract the row_df of the best configuration + RMSE
    curr_best_conf <- curr_xgb_tune$results[rownames(curr_xgb_tune$bestTune),c(1:8)]
    curr_best_model <- curr_xgb_tune$finalModel
    
    # append new row to final_confs_df
    new_row_data <- c(wsid_num)
    new_row_data <- append(new_row_data,curr_best_conf)
    new_row <-data.frame(new_row_data)
    names(new_row)<- cols_names
    final_confs_df <- rbind(final_confs_df, new_row)
    
    # save the trained models
    if(save_models){
      model_RMSE <- round(curr_best_conf$RMSE, digits = 4)
      model_name <- paste(wsid_num,'_XGB_model_',model_RMSE,'.model',sep='')
      xgb.save(curr_best_model,paste(path_for_models,model_name,sep='/'))
      print(paste('Model:',wsid_num,' has been saved'))
    }
    
  }
  
  # save the best conf df as a csv
  write.csv(final_confs_df,paste(path_for_csv,'final_confs_df.csv',sep='/'),row.names = FALSE)
  
}

csv_path <- models_optimization_path
models_path <- models_optimization_path

# sub_data <- subset(weather_data_processed, weather_data_processed$wsid %in% c(178,309,310,314))

XGB_randomSearch(weather_data_final, # <-- Enter the processed data (can be subset of the all_data)
                 path_for_csv=csv_path, # <-- Path for saving the csv with the best configurations
                 save_models=T, # <-- Set True if you want to save the final trained models
                 path_for_models=models_path, # <-- Path for saving the final trained models
                 nrounds=8000, # <-- choose how many epochs to do in each fold 
                 n_max_depth=3, # <-- choose how many values of max_depth to check
                 n_eta=2, # <-- choose how many values of n_eta to check
                 n_gamma=2, # <-- choose how many values of n_gamma to check
                 n_subsample=1) # <-- choose how many values of n_subsample to check

