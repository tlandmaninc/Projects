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

#setting current working directory of the code as the project folder.So that we don't have to keep changing every time.
project_path <- setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Create Data directory
data_path <- paste(project_path,"Data", sep = "/") 
if(!dir.exists(data_path))
  dir.create(data_path)
weather_data_path <- paste(data_path,"train_data.csv", sep = "/")

weather_traindata <- as.data.frame(fread(weather_data_path))

weather_testdata_path <- paste(data_path,"test_data.csv", sep = "/")

ref_stations <- list()

#Create Stations directory
ws_data_path <- paste(project_path, "Data/Stations/", sep ="/")
if(!dir.exists(ws_data_path))
  dir.create(ws_data_path)
#Create Models directory
XGB_models_path <- paste(project_path, "NEWTESTMODELS_Normalized", sep ="/")
if(!dir.exists(XGB_models_path))
  dir.create(XGB_models_path)

models_optimization_path <- paste(project_path, "Optimization/", sep ="/")
if(!dir.exists(models_optimization_path))
  dir.create(models_optimization_path)

models_list <- list()
file.names <- dir(XGB_models_path, pattern =".Rds")
for(i in 1:length(file.names)){
  full_path <- file.path(XGB_models_path,file.names[i])
  
  loaded_model <- readRDS(full_path)
  
  # Extracting the wsid from the model file
  # assuming the file name is in the following format exp: 'Station_178_RMSE_13.3341545207481.Rds'
  if(!(str_split(file.names[i],'_')[[1]][1] %in% "Best")){
  model_number <- str_split(file.names[i], '_')[[1]][2]
  models_list[[model_number]] <- loaded_model
  }
}

##################################################
####        Utility Functions           ####      
##################################################

#3. Euclidean distance calculation function :
calc_Euclidean_distance <- function(lat_1, lon_1, lat_2, lon_2){
  result <- sqrt((lat_1 - lat_2)**2 + (lon_1 - lon_2)**2)
  return(result)
}

#2. Get closest WSID by lat long euclidean distance:
get_closest_wsid <- function(spatial_info,lat,lon){
  known_wsid <- spatial_info$wsid
  distance_vector <- c()
  
  for(wsid_num in known_wsid){
    current_lat <- subset(spatial_info, spatial_info$wsid==wsid_num)$lat
    current_lon <- subset(spatial_info, spatial_info$wsid==wsid_num)$lon

    distance <- calc_Euclidean_distance(current_lat,current_lon,lat,lon)

    distance_vector <- c(distance_vector,distance)

  }
  
  min_index <- which.min(distance_vector)
  closest_wsid <- known_wsid[min_index]
  
  return(closest_wsid)
}

#1. Get reference stations for new wsids
set_reference_wsids <- function(test_data, train_data){
  
  spatial_info <- data.frame(matrix(ncol = 5, nrow = 0))
  x <- c("wsid", "lat", "lon","city","prov")
  colnames(spatial_info) <- x
  
  
  for (wsid_num in unique(train_data$wsid)){
    wsid_row <- subset(train_data, train_data$wsid==wsid_num)[1,]
    
    new_row <- data.frame(wsid_num,wsid_row$lat,wsid_row$lon,wsid_row$city,wsid_row$prov)
    colnames(new_row) <- x
    spatial_info <- rbind(spatial_info,new_row)
  }
  
  print(spatial_info)
  
  
  for (wsid_num in unique(test_data$wsid)){
    wsid_subset <- subset(test_data,test_data$wsid == wsid_num)
    
    # according to the ref station we will replace Nan and choose model
    ref_station <- 0 
    
    # check if this wsid is known from training
    if(wsid_num %in% train_data$wsid){
      ref_station <- wsid_num
    }
    else{ # find the closet station
      #cat("Current working dir: ", wsid_num)
      test_lat <- wsid_subset$lat[1]
      test_lon <- wsid_subset$lon[1]
      ref_station <- get_closest_wsid(spatial_info,test_lat,test_lon)
    }
    
    # save the ref stations for all the test stations
    ref_stations[[wsid_num]] <- ref_station
    cat("Test Station : ",wsid_num,"Matched Station : ", ref_stations[[wsid_num]])
    
  }
  return(ref_stations)
}

##################################################
####        Preprocessing Functions           ####      
##################################################

# 1. drop irrelevant columns
drop_irrelevant_cols <- function(df, cols_to_drop){
  df <- df[,!(names(df) %in% cols_to_drop)]  
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
  df <- subset(df, df$gbrd<9000) #< 4500
  nrows_after = nrow(df)
  print(paste(" 2.3",(nrows_before-nrows_after), 'anomalous records of gbrd were removed'), quote=FALSE)
  return(df)
}

# Adds a new part of the day feature to data
createPartOfDayFeature <- function(df){
  # 0 - Morning, 1 - Afternoon, 2 - Evening, 3 - Night
  df$part_of_day<- as.factor(ifelse(df$hr>=6 & df$hr<=11, 0, 
                                    ifelse(df$hr>=12 & df$hr<=17, 1,
                                           ifelse (df$hr>=18 & df$hr<=23,2,3))))
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
  df$day_of_week<-as.factor(strftime(as.Date(df$date, "%Y-%m-%d"), "%u"))
  
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

## 5. Missing Values Completion using Spline Interpolation
missingValuesCompletion <- function(df){
  # Convert mdct column to date POSIXct expression
  df$mdct <- as.POSIXct(df$mdct, format = "%Y-%m-%d %H:%M", tc="GMT")

  # Removing NAs from mdct
  df_sliced <-subset(df, !is.na(df$mdct))
  # df_as_ts <- as.xts(wsid_data_p, order.by=wsid_data_p$mdct)

  # Complete missing values using Spline Interpolation
  interpolated_df <- as.data.table(na_interpolation(df_sliced,option= 'spline'))
  print("5. Missing values completion using Spline Interpolation was performed on gbrd & gust columns", quote=FALSE)
  return(interpolated_df)
}

## 7. Arranges data so the temperature variable will be placed in the last column
organize_data <- function(df){
  col_order <- c(seq(1,9),seq(11,ncol(df)),10)
  organized_df <- df[,col_order, with=FALSE]
  print("7. Data was organized successfully!", quote=FALSE)
  return(organized_df)
}


##################################################
####    Test Data Preprocessing Procedure    ####      
##################################################

preprocessingTestData <- function(df){
  
  ######## 1. Drops irrelevant columns ########
  cols_to_drop <- c('wsnm','elvt','lat','lon','inme','city','prov','wdsp', 'prcp') # 'gbrd', 'date', 'mdct'
  df_sliced <- drop_irrelevant_cols(df, cols_to_drop)
  
  weather_traindata_sliced <-drop_irrelevant_cols(weather_traindata, cols_to_drop)
  
  ######## 2. Records to be removed ########
  #df_removed <- removingRecordsProcedure(dfSliced)
  
  ######## 3. Feature Engineering ########
  df_removed <- featureEngineeringProcedure(df_sliced) 
  weather_traindata_removed <- featureEngineeringProcedure(weather_traindata_sliced)
  ######## 4. Factoring Columns ########
  discrete_colums <- c('yr', 'mo', 'da', 'hr')
  df_removed <- cols_to_factors(df_removed, discrete_colums)
  weather_traindata_removed$day_of_week<-as.factor(strftime(as.Date(weather_traindata_removed$date, "%Y-%m-%d"), "%u"))
  weather_traindata_removed <- createPartOfDayFeature(weather_traindata_removed)
  
  weather_traindata_removed$yr <- factor(weather_traindata_removed$yr)
  weather_traindata_removed$mo <- factor(weather_traindata_removed$mo)
  weather_traindata_removed$da <- factor(weather_traindata_removed$da)
  weather_traindata_removed$hr <- factor(weather_traindata_removed$hr)

  levels(df_removed$part_of_day) <- levels(weather_traindata_removed$part_of_day)

  levels(df_removed$day_of_week) <- levels(weather_traindata_removed$day_of_week)
  
  levels(df_removed$yr) <- levels(weather_traindata_removed$yr)
  
  levels(df_removed$mo) <- levels(weather_traindata_removed$mo)
  
  levels(df_removed$da) <- levels(weather_traindata_removed$da)
  
  levels(df_removed$hr) <- levels(weather_traindata_removed$hr)
  
  ######## 5. Missing Values Completion using Spline Interpolation ########
  df_interpolated <- missingValuesCompletion(df_removed)
  
  ######## 6. Drops columns that were required for feature Engineering phase ########
  df_interpolated<- subset(df_interpolated,select=-c(mdct,date))
  print("6. mdct & date columns were dropped", quote=FALSE)
  
  ######## 7. Organize columns - move temp to the end of data table  ########
  current_weather_station_final <- organize_data(df_interpolated)
  
  return(as.data.table(current_weather_station_final))
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

##################################################
####   Testing an XGB per weather station     ####      
##################################################

testEnsembleOfXGBs <- function(test_data,models,final_predictions,rf_st){
  for(test_wsid in unique(test_data$wsid)){
    # get the reference station 
    ref_station <- rf_st[[test_wsid]]
    # get the relevant model for predictions (some how its works only this way)
    ref_model <- models_list[toString(ref_station)]
    
    # prepare the data for predictions
    sub_test_data <- test_data[test_data$wsid == test_wsid,]
    
    print(paste('Predicting test_wsid:', test_wsid, 'ref model:', ref_station))
    #print(str(sub_test_data))
    sub_id_list <- sub_test_data$id
    dropper <- c("id","wsid")
    
    
    sub_test_data$hmin <- as.integer(sub_test_data$hmin)
    
    #print(str(sub_test_data))
    col_order <- c("yr", "mo", "da",
                   "stp", "smax","smin","gbrd","dewp","dmax","dmin","hmdy","hmax"
                   ,"hmin","wdct","gust","s_diff","dew_diff","h_diff","day_of_week"
                   ,"part_of_day","id","wsid")
    sub_test_data_reordered <- sub_test_data[, col_order]
    
    #fwrite(sub_test_data_reordered, "testFinalBeforePreds.csv")
    
    
    #print(str(sub_test_data))
    
    
    prepared_test_data <- model.matrix(~.+0,data = subset(sub_test_data_reordered,select=-c(id,wsid))) 
    #print(names(prepared_test_data))
    #print(prepared_test_data)
    
    # predict the temp
    sub_predictions <- predict(ref_model,prepared_test_data)[[1]]
    
    # convert the results to a df in the right format
    predictions_df <- data.frame(sub_id_list,sub_predictions)
    
    colnames(predictions_df) <- c('id','temp')
    final_predictions <- rbind(final_predictions,predictions_df)
    
  }
  
  return(final_predictions)
  
}

# Executes Train Pipeline 
executeTestDataPipeline <- function(){
  
  ##################################################
  ####              Loading Data                ####      
  ##################################################
  
  print("Loading Data....")
  
  # load data into a data frame object
  weather_data <- as.data.frame(fread(weather_testdata_path))
  ref_stations = set_reference_wsids(weather_data, weather_traindata)
  
  print("Data was loaded successfully!")
  
  ##################################################
  ####          Data Preprocessing              ####      
  ##################################################
  
  print("Preprocessing Data....")
  
  # Preprocess whole data
  weather_testdata_processed <- preprocessingTestData(weather_data)
  
  # Print Summary
  print(paste("# Weather data was preprocessed succesfully and ended up with",nrow(weather_testdata_processed), "records."), quote=FALSE)  
  
  # Create vector of unique weather stations' IDs
  unique_wsid <- sort(unique(weather_testdata_processed$wsid))
  
  
  ##################################################
  ####       Explanatory Data Analysis          ####      
  ##################################################
  
  # Convert data to data frame
  #fwrite(weather_testdata_processed, "testFinalBeforeDF.csv")
  weather_testprocessed_df <- as.data.frame(weather_testdata_processed)
  
  # Extract Continuous features
  #continuos_data <- weather_processed_df[,sapply(weather_processed_df, is.numeric)]
  #continuos_data <- drop_useless_records(continuos_data, seq(1,ncol(continuos_data)))
  
  # Calculate Pearson Correlation matrix
  #pearson_corr_matrix <- cor(continuos_data, method = "pearson")
  
  # Visualize Pearson Correlation for Continuous Data
  #corrplot(pearson_corr_matrix, method ="number")
  
  # Extract Nominal features
  #nominal_data <- weather_processed_df[,sapply(weather_processed_df, is.factor)]
  
  # CramerV Correlation between Hour (hr) Vs. part_of_day - CramerV of 100%
  #cramerV(nominal_data$hr,nominal_data$part_of_day)
  
  # Remove hr
  correlated_features_to_be_removed <- c("hr")
  weather_testdata_final <- drop_irrelevant_cols(weather_testprocessed_df, correlated_features_to_be_removed)
  
  #fwrite(weather_testdata_final, "testFinal.csv")
  #ref_stations = set_reference_wsids(weather_testdata_final, weather_traindata)
  #print("REF STATIONS :::::")
  #print(ref_stations)
  
  ##################################################
  ####            Testing Procedure             ####      
  ##################################################
  final_predictions <- data.frame(matrix(ncol = 2, nrow = 0))
  x <- c("id", "temp")
  colnames(final_predictions) <- x
  
  # create predictions dataframe
  final_predictions <- testEnsembleOfXGBs(weather_testdata_final, models_list , final_predictions,ref_stations)
  save_path = paste(project_path,"predictions_testPIPE.csv", sep = "/") 
  write.csv(final_predictions,save_path, row.names = FALSE)

  return(weather_testdata_final)
}

##################################################
####        Executes Test Pipeline            ####      
##################################################

weather_data_final = executeTestDataPipeline()

