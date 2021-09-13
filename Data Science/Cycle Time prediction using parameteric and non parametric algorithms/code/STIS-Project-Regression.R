# Packages to be installed
#install.packages('Metrics')
#install.packages("DescTools")
#install.packages('qpcR')

# Importing Packages
library(data.table)
library(DescTools)
library(caret)
library(qpcR)
library(boot)
##################################################
####              Loading Data                ####      
##################################################

print("Loading Data....")

data_path <- "D:/MSc\ Degree/Courses/2nd\ Year/6.\ Selected\ Topics\ In\ Statistics/Project/Data/Final_Data.csv"

# load data into a data frame object
df <- as.data.frame(fread(data_path))
print("Data was loaded successfully!")

# Convert discrete colums to factors
df$`Lot calsification` <- as.factor(df$`Lot calsification`) 
df$LOT_TYPE_Clean <- as.factor(df$LOT_TYPE_Clean) 
df$Machine <- as.factor(df$Machine) 
df$Job <- as.factor(df$Job) 
df$LAST_LOT <- as.factor(df$LAST_LOT)

set.seed(1111)

# defining training control 
# as cross-validation and  
# value of K equal to 10 
train_control <- trainControl(method = "cv", 
                              number = 10) 
# training the model by assigning sales column 
# as target variable and rest other column 
# as independent varaible 
LR_basic_MAE <- train(`Process duration`~., data = df,  
               method = "lm", metric = "MAE",
               trControl = train_control) 
print(LR_basic_MAE)

LR_basic_MSE <- train(`Process duration`~., data = df,  
                      method = "lm", metric = "AIC",
                      trControl = train_control) 
print(LR_basic_MSE)

# printing model performance metrics 
# along with other details 
print(LR_basic_MAE)

# Basic Linear Regression Model & Summary
LR_basic <- lm(`Process duration`~., data=df)

# Print Summary - Basic Model
summary (LR_basic)

MAE <- as.character(round(DescTools::MAE(LR_basic),3))
MSE <- as.character(round(MSE(LR_basic),3))
R_Squared <- as.character(round(summary(LR_basic)$r.squared,3))
AIC <- as.character(round(-1 * AIC(LR_basic),3))
BIC <- as.character(round(-1 * BIC(LR_basic),3))
metrics_vector <- paste("Regression Metrics:\nMAE: ", MAE, ", MSE: ", MSE, 
                        ", R Squared: ",R_Squared, "AIC: ", AIC, ", BIC: ", BIC)
cat(metrics_vector)


# Forward Selection
min.LR_basic_model = lm(log(`Process duration`)~1, data=df)
max.LR_basic_model <- formula(lm(log(`Process duration`)~., df))
fwd.LR_basic = step(min.LR_basic_model, direction='forward', scope=max.LR_basic_model)
fwd.LR_basic$anova
summary(fwd.LR_basic)

# Backward Selection
bwd.LR_basic = step(LR_basic, direction='backward')
bwd.LR_basic$anova
summary(bwd.LR_basic)

# Stepwise Selection
stepwise.LR_basic = step(min.LR_basic_model, direction='both', scope=max.LR_basic_model)
stepwise.LR_basic$anova
summary(stepwise.LR_basic)

# Print Forward selection metrics - Unified for all feature selection methods
MAE <- as.character(round(DescTools::MAE(fwd.LR_basic),3))
MSE <- as.character(round(MSE(fwd.LR_basic),3))
R_Squared <- as.character(round(summary(fwd.LR_basic)$r.squared,3))
AIC <- as.character(round(-1 * AIC(fwd.LR_basic),3))
BIC <- as.character(round(-1 * BIC(fwd.LR_basic),3))
metrics_vector <- paste("Forward Selection Regression Metrics:\nMAE: ", MAE, ", MSE: ", MSE, 
                        ", R Squared: ",R_Squared, "AIC: ", AIC, ", BIC: ", BIC)
cat(metrics_vector)


# Checking Model's Assumptions

# 1. Linearity Assumption

# Get list of residuals and fitted values
Residuals <- resid(fwd.LR_basic)
Fitted <- fitted(fwd.LR_basic)

# Calculate Max residual
max(Residuals)
# Calculate Min residual
min(Residuals)

# Produce residual vs. fitted plot
plot(Fitted, Residuals)
title("Residuals vs. Fitted")
abline(0,0) #add a horizontal line at 0 

# Produce residual vs. fitted plot
plot(fwd.LR_basic, 3)
title("Standardized Residuals vs. Fitted")
abline(0,0)


# 2. Homoscedasticity Assumption
anova()
model2 <- lm(log(`Process duration`)~., data = df)
plot(model2, 3)


# 3. Normality Assumption

# Produce QQ-plot
Standardized_Residuals <- (Residuals - mean(Residuals)) / sqrt(var(Residuals))
qqnorm(Standardized_Residuals)
abline(a=0, b=1)

# One-sample Kolmogorov-Smirnov test
ks.test(x=Standardized_Residuals,y="pnorm",alternative = "two.sided", exact = NULL)

# Shapiro-Wilk normality test
shapiro.test(Standardized_Residuals)

# Residuals Histogram
hist(Standardized_Residuals, xlab ="Normalized error", main="Histogram of normalized error")

hist(log(Standardized_Residuals), xlab ="Normalized error", main="Histogram of the log of normalized error")

################################## Transformations #############################

# Forward Selection using Box-Cox log transformation on the dependent variable
min.LR_log_model = lm(log(`Process duration`)~1, data=df)
max.LR_log_model <- formula(lm(log(`Process duration`)~., df))
fwd.LR_log = step(min.LR_basic_model, direction='forward', scope=max.LR_basic_model)
fwd.LR_log$anova
summary(fwd.LR_log)


# Forward Selection with polynomial transformation and Box-Cox transformation on features
min.LR_polynomial_model = lm(log(`Process duration`)~1, data=df)
max.LR_polynomial_model <- formula(lm(log(`Process duration`)~ poly(PARTIALITY_SCORE,2)+
                          poly(OVERLAP_SECONDS,2)+ poly(PROCESSED_WAFER_COUNT,2)
                          + poly(`Wafer STDV`, 2)+`Lot calsification` + Job
                          + Machine + LAST_LOT,df))

fwd.LR_polynomial = step(min.LR_polynomial_model, direction='forward', 
                         scope=max.LR_polynomial_model)

# Print Forward selection on Polynomial Regression
fwd.LR_polynomial$anova
summary(fwd.LR_polynomial)


# Checking Polynomial Model's Assumptions

# 1. Linearity Assumption & 2. Homoscedasticity Assumption

# Get list of residuals and fitted values
Poly_Residuals <- resid(fwd.LR_polynomial)
Poly_Fitted <- fitted(fwd.LR_polynomial)

# Calculate Max residual
max(fwd.LR_polynomial)
# Calculate Min residual
min(fwd.LR_polynomial)

# Produce residual vs. fitted plot
plot(Poly_Fitted, Poly_Residuals)
title("Residuals vs. Fitted")
abline(0,0) #add a horizontal line at 0 

# Produce residual vs. fitted plot
plot(fwd.LR_polynomial, 3)
title("Standardized Residuals vs. Fitted")
abline(0,0)

# 3. Normality Assumption

# Produce QQ-plot
Standardized_Poly_Residuals <- (exp(Poly_Residuals) - mean(exp(Poly_Residuals))) / sqrt(var(exp(Poly_Residuals)))
qqnorm(Standardized_Poly_Residuals)
abline(a=0, b=1)

# One-sample Kolmogorov-Smirnov test
ks.test(x=Standardized_Poly_Residuals,y="pnorm",alternative = "two.sided", exact = NULL)

# Residuals Histogram
hist(Standardized_Poly_Residuals, xlab ="Normalized error", main="Histogram of normalized error")
hist(log(Standardized_Poly_Residuals), xlab ="Normalized error", main="Histogram of the log of normalized error")

# Print Forward selection metrics - Unified for all feature selection methods
MAE <- as.character(round(DescTools::MAE(fwd.LR_polynomial),3))
MSE <- as.character(round(MSE(fwd.LR_polynomial),3))
R_Squared <- as.character(round(summary(fwd.LR_polynomial)$r.squared,3))
AIC <- as.character(round(-1 * AIC(fwd.LR_polynomial),3))
BIC <- as.character(round(-1 * BIC(fwd.LR_polynomial),3))
metrics_vector <- paste("Forward Selection Regression Metrics:\nMAE: ", MAE, ", MSE: ", MSE, 
                        ", R Squared: ",R_Squared, "AIC: ", AIC, ", BIC: ", BIC)
cat(metrics_vector)



# GLM - Exponential Regression
# defining training control 
# as cross-validation and  
# value of K equal to 10 
exp.glm <- glm(log(`Process duration`) ~., data = df)
cv.glm(data = df, exp.glm, K=10)


train_control <- trainControl(method = "cv", 
                              number = 10) 
# training the model by assigning sales column 
# as target variable and rest other column 
# as independent varaible 
LR_basic_MAE <- train(`Process duration`~., data = df,  
                      method = "glm", metric = "MAE",
                      trControl = train_control) 
print(LR_basic_MAE)

LR_basic_MSE <- train(`Process duration`~., data = df,  
                      method = "lm", metric = "AIC",
                      trControl = train_control) 
print(LR_basic_MSE)

# printing model performance metrics 
# along with other details 
print(LR_basic_MAE)

# Basic Linear Regression Model & Summary
LR_basic <- lm(`Process duration`~., data=df)

# Print Summary - Basic Model
summary (LR_basic)

MAE <- as.character(round(DescTools::MAE(LR_basic),3))
MSE <- as.character(round(MSE(LR_basic),3))
R_Squared <- as.character(round(summary(LR_basic)$r.squared,3))
AIC <- as.character(round(-1 * AIC(LR_basic),3))
BIC <- as.character(round(-1 * BIC(LR_basic),3))
metrics_vector <- paste("Regression Metrics:\nMAE: ", MAE, ", MSE: ", MSE, 
                        ", R Squared: ",R_Squared, "AIC: ", AIC, ", BIC: ", BIC)
cat(metrics_vector)

