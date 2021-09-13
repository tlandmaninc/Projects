library(readxl)
library(dplyr)
library(plyr)
library(sandwich)
library(nnet)
library(gridExtra)
library(MASS)
library(ggplot2)


####################################### Logistic Regression ##################################
  
pathTr<-"TitanicTrain.csv"
pathTe<-"TitanicTest.csv"

#Titanic Data
TitanicTrain<-read.csv(pathTr,header = TRUE, sep = ',',colClasses = 
                         c("factor","factor","factor","factor","factor",
                           "factor","factor","factor"))

TitanicTest<-read.csv(pathTe,header = TRUE, sep = ',',colClasses = 
                        c("factor","factor","factor","factor","factor",
                          "factor","factor","factor"))

# Converting -1 to 0
TitanicTrain$Survived<-as.numeric(TitanicTrain$Survived)
TitanicTrain[which(TitanicTrain[,"Survived"]==2),c("Survived")] <- 0
TitanicTrain$Survived <- as.factor(TitanicTrain$Survived)


TitanicTest$Survived<-as.numeric(TitanicTest$Survived)
TitanicTest[which(TitanicTest[,"Survived"]==2),c("Survived")] <- 0
TitanicTest$Survived <- as.factor(TitanicTest$Survived)


TitanicTest <- TitanicTest[,-9]


# Summary
summary(TitanicTrain)

# Logistic Model Training
set.seed(1234)
LogisticRegMod1<-glm(TitanicTrain$Survived~.,data = TitanicTrain,family = "binomial")

# Model Summary
summary(LogisticRegMod1)

#Predict on Train Data
PredictionsProb<-predict(LogisticRegMod1,newdata=TitanicTest[,-1])
Predictions<-as.factor(ifelse(PredictionsProb > 0.0001,1,0))

#Building Confusion Matrix on Train Data
confusionMat<-table(Predictions,"Actual"=TitanicTest$Survived)
confusionMat

#Calculating Accuracy
Accuracy<-sum(diag(confusionMat))/sum(confusionMat)
Accuracy
#91 %


####################################### Poisson Regression ##################################

# Number of awards earned by students at one high school. Predictors of the number of awards 
# earned include the type of program in which the student was enrolled 
# (e.g., vocational, general or academic) and the score on their final exam in math.

path<-"poisson_sim.csv"

#Student's Awards Data
studentsData<-read.csv(path,header = TRUE, sep = '|')

#Converting prog to Factor
studentsData$prog<-as.factor(studentsData$prog)

#Removing ID
studentsData<-studentsData[,-1]

summary(studentsData)

#Group Data by Awards
AwardsCounts<-studentsData %>% group_by(Value=studentsData$num_awards) %>% count()
AwardsCounts$n<-as.factor(AwardsCounts$n)

# num_awards Histogram
hist(studentsData$num_awards,col =  "orange")

#Group Data by Prog 
statProg<-ddply(studentsData,"prog",summarise,
                counts=sum(num_awards>=0),
                meanAwards=mean(num_awards),
                varAwards=var(num_awards),
                varByMean=varAwards/meanAwards)
statProg

# Mild overdispersion on Academic =>  ~ 1.635
# Small underdispersion on General =>  ~ 0.819      => Heteroskedasticity
# Small overdispersion on Vocational =>  ~ 1.116

#Splitting Data to Train & Test sets (No validation set to simplify)
# Split into train [80%] & test [20%] sets
set.seed(1234)
Train_id  <- sample(1:nrow(studentsData), 0.8*nrow(studentsData))
TrainData <- as.data.frame(studentsData[Train_id,])
TestData  <- as.data.frame(studentsData[-Train_id,])
rm(Train_id)

# Poisson Model Training
set.seed(1234)

PoissonRegMod1<-glm(num_awards~.,data = TrainData,family = "poisson")

# Model Summary
summary(PoissonRegMod1)

# Multiplicative Effects of predictors
exp(PoissonRegMod1$coefficients)

#Little bit of Skewness on the Deviance Residuals (Median = -0.5106)
# => Applying Robbost method to reduce std error

#Cameron & Trivedi (2009) 
#recommended using robust standard errors for the parameter estimates to control for 
# mild violation of the distribution assumption that  variance = mean

# Dealing with Heteroskedasticity
cov.output<-vcovHC(PoissonRegMod1,type="HC0")


#Calculating Std Erors
stdError<-sqrt(diag(cov.output))

#Create Robust Estimates Summary Table
robust<-cbind(Estimate=coef(PoissonRegMod1),
              "Robust Std Error"=stdError,
              "Robust Z value"=coef(PoissonRegMod1)/stdError,
              "Pr(>|Z|)"=2*pnorm(abs(coef(PoissonRegMod1)/stdError),lower.tail = FALSE),
              LL=coef(PoissonRegMod1)-1.96*stdError,
              UL=coef(PoissonRegMod1)+1.96*stdError)


#Compare between Std. Err & P.vals
coef(summary(PoissonRegMod1))          
robust

#Std Error reduced & P values are significant (0.05)   

# Check if the model is a good fit or not - LRT Chi Square test
with(PoissonRegMod1,
     cbind(res.deviance=deviance,df=df.residual,chi=(null.deviance-deviance),
     p=pchisq(null.deviance-deviance,df.null-df.residual,lower.tail = FALSE)))
               
#  => There is a strong evidence that The residuals Model 
# fits better the data than the NULL Model in 0.05 %

#The null deviance shows how well the response variable is predicted by 
#a model that includes only the intercept (grand mean)



#Predict on Test Data
PoisPredictions<-predict(PoissonRegMod1,newdata = TestData[,-1],type = "response")

#Building Confusion Matrix on Test Data
confusionMatPois<-table(ceiling(PoisPredictions),TestData$num_awards)
confusionMatPois

#Calculating Accuracy
Accuracy<-sum(diag(confusionMatPois))/sum(confusionMatPois)
Accuracy
#70 %

#############################################################################################
 


##################################### Multinomial Regression ################################

irisData<-iris

summary(irisData)
#Balanced Class => c(50,50,50)

# Correlation Plot
plot(col=factor(irisData$Species),irisData[,1:4])



#Splitting Data to Train & Test sets (No validation set to simplify)
# Split into train [80%] & test [20%] sets
set.seed(1234)
Train_id  <- sample(1:nrow(irisData), 0.8*nrow(irisData))
TrainData <- irisData[Train_id,]
TestData  <- irisData[-Train_id,]
rm(Train_id)
summary(TestData)


#Multinomial Regression Model - On Train Data
set.seed(1234)
MultinomMod1<-multinom(factor(TrainData$Species)~.,data=TrainData)
summary(MultinomMod1)  

#Predict on Test Data
predictions<-predict(MultinomMod1,newdata=TestData[,1:4])

#Building Confusion Matrix
confusionMat<-table(predictions,TestData[,5])
confusionMat


#Calculating Accuracy on Test- Relevent because specias is balanced
Accuracy<-sum(diag(confusionMat))/sum(confusionMat)
Accuracy
# ~ 97 %


#Plot Predicted Values
PredictedPlot<-ggplot(TestData,aes(Sepal.Length,Sepal.Width))+
  geom_point(aes(colour = predictions)) + ggtitle("Predicted",)

#Plot Actual Values
ActualPlot<-ggplot(TestData,aes(Sepal.Length,Sepal.Width))+
  geom_point(aes(colour =Species)) + ggtitle("Actual")


#Create Plots Grid
grid.arrange(PredictedPlot,ActualPlot,ncol=1)





############################################################################################

###################################### Ordinal Regression ##################################

#2126 fetal cardiotocograms (CTGs) were automatically processed and the respective 
#diagnostic features measured. The CTGs were also classified by three expert obstetricians 
#and a consensus classification label assigned to each of them. Classification was with 
#to a fetal state (N, S, P). 

#Loading dataset from xls file
Cardiotocographic <- read_excel("CTG.xls", sheet="Data",col_names = TRUE)

#Coverting class to factor
Cardiotocographic$CLASS<-as.factor(Cardiotocographic$CLASS)

#Coverting NSP to ordered
Cardiotocographic$NSP<-as.ordered(Cardiotocographic$NSP)


summary(Cardiotocographic)

#Splitting Data to Train & Test sets (No validation set to simplify)

# Split into train [80%] & test [20%] sets
set.seed(1234)
Train_id  <- sample(1:nrow(Cardiotocographic), 0.8*nrow(Cardiotocographic))
TrainData <- as.data.frame(Cardiotocographic[Train_id,])
TestData  <- as.data.frame(Cardiotocographic[-Train_id,])
rm(Train_id)

# Ordinal Regression Model
set.seed(1234)
OrdinalMod1<-polr(NSP~LB+AC+FM+DL+DS+DP,data = TrainData,Hess = TRUE)
OrdinalMod1

# LB	baseline value 
# AC	accelerations 
# FM	foetal movement 
# DL	light decelerations
# DS	severe decelerations
# DP	prolongued decelerations
# DR	repetitive decelerations


summary(OrdinalMod1)

#Adding p-values
coefficients<-coef(summary(OrdinalMod1))
DetailedSummary<-cbind(coefficients,"p value"=pnorm(abs(coefficients[,"t value"]),lower.tail = FALSE)*2)
DetailedSummary


#Predict on Test Data
predictions<-predict(OrdinalMod1,newdata = TestData)

#Building Confusion Matrix
confusionMat<-table(predictions,"Actual"=TestData$NSP)
confusionMat


#Calculating Accuracy on Test
Accuracy<-sum(diag(confusionMat))/sum(confusionMat)
Accuracy
# ~ 80.3 %
