################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 4: Over-Fitting and Model Tuning
###
### Required packages: caret, doMC (optional), kernlab
###
### Data used: 
###
### Notes: 
### 1) This code is provided without warranty.
###
### 2) This code should help the user reproduce the results in the
### text. There will be differences between this code and what is is
### the computing section. For example, the computing sections show
### how the source functions work (e.g. randomForest() or plsr()),
### which were not directly used when creating the book. Also, there may be 
### syntax differences that occur over time as packages evolve. These files 
### will reflect those changes.
###
### 3) In some cases, the calculations in the book were run in 
### parallel. The sub-processes may reset the random number seed.
### Your results may slightly vary.
###
################################################################################

### Section 4.1 The Problem of Over-Fitting (No code!) ####
### Section 4.2 Model Tuning (No code!) ####
### Section 4.3 Data Splitting (No code!) ####
### Section 4.4 Resampling Techniques (No code!) ####
### Section 4.5 Case Study: Credit Scoring (No code!) ####

### Section 4.6 Choosing Final Tuning Parameters ####

library(caret)
data(GermanCredit)
help(GermanCredit)

## First, remove near-zero variance (變異數、方差) predictors then get rid of a few predictors 
## that duplicate values. For example, there are two possible values for the 
## housing variable: "Rent", "Own" and "ForFree". So that we don't have linear
## dependencies, we get rid of one of the levels (e.g. "ForFree")

summary(GermanCredit)

## Check the ferquency distribution of each binary variable
sapply(names(GermanCredit)[8:length(GermanCredit)], function(u) prop.table(table(GermanCredit[,u])))
names(GermanCredit)[nearZeroVar(GermanCredit)]

GermanCredit <- GermanCredit[, -nearZeroVar(GermanCredit)] # 62 - 13 -> 49 variables
names(GermanCredit)

## remove CheckingAccountStatus.lt.0, SavingsAccountBonds.lt.100, EmploymentDuration.lt.1, EmploymentDuration.Unemployed, Personal.Male.Married.Widowed, Property.Unknown, and Housing.ForFree (Why? Not so even again!)
## Check the ferquency distribution of each variable again
sapply(names(GermanCredit)[8:length(GermanCredit)], function(u) prop.table(table(GermanCredit[,u])))
sapply(c("CheckingAccountStatus.lt.0","SavingsAccountBonds.lt.100","EmploymentDuration.lt.1","EmploymentDuration.Unemployed","Personal.Male.Married.Widowed","Property.Unknown","Housing.ForFree"), function(u) prop.table(table(GermanCredit[,u])))
# GermanCredit$CheckingAccountStatus.lt.0 <- NULL
# GermanCredit$SavingsAccountBonds.lt.100 <- NULL
# GermanCredit$EmploymentDuration.lt.1 <- NULL
# GermanCredit$EmploymentDuration.Unemployed <- NULL
# GermanCredit$Personal.Male.Married.Widowed <- NULL
# GermanCredit$Property.Unknown <- NULL
# GermanCredit$Housing.ForFree <- NULL
# # 47 -> 40 variables

## Split the data into training (80%) and test sets (20%)
set.seed(100)
?createDataPartition
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]] # A (named) list or matrix of row position integers corresponding to the training data
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]

?sample
set.seed(1)
tr_idx <- sample(1:1000, 800)
GermanCreditTraining <- GermanCredit[tr_idx,]
GermanCreditTesting <- GermanCredit[-tr_idx,]

## The model fitting code shown in the computing section is fairly
## simplistic.  For the text we estimate the tuning parameter grid
## up-front and pass it in explicitly. This generally is not needed,
## but was used here so that we could trim the cost values to a
## presentable range and to re-use later with different resampling
## methods.

library(kernlab)
set.seed(231)
?sigest # Hyperparameter estimation for the Gaussian Radial Basis kernel (frac: Fraction of data to use for estimation. By default a quarter of the data (0.5) is used to estimate the range of the sigma hyperparameter.)
sigDist <- sigest(Class ~ ., data = GermanCreditTrain, frac = 1) # Returns a vector of length 3 defining the range (0.1 quantile, median and 0.9 quantile) of the sigma hyperparameter.
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2^(-2:7)) # Use the 0.1 quantile as sigma

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.

### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the available hardware. We
### estimate the memory usage (VSIZE = total memory size) to be 
### 2566M/core.

library(doMC)
getDoParWorkers()
registerDoMC(4)
getDoParWorkers()

# load("all_04.RData")

set.seed(1056)
system.time(svmFit <- train(Class ~ .,
                data = GermanCreditTrain,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "repeatedcv", 
                                         repeats = 5,
                                         classProbs = TRUE)))
## classProbs = TRUE was added since the text was written
## user  system elapsed 
## 289.552   1.574  29.771 

## Print the results
svmFit

## A line plot of the average performance. The 'scales' argument is actually an 
## argument to xyplot that converts the x-axis to log-2 units.

plot(svmFit, scales = list(x = list(log = 2)))

## Test set predictions

predictedClasses <- predict(svmFit, GermanCreditTest)
str(predictedClasses)

## Use the "type" option to get class probabilities

predictedProbs <- predict(svmFit, newdata = GermanCreditTest, type = "prob")
head(predictedProbs)


## Fit the same model using different resampling methods. The main syntax change
## is the ***control*** object.

set.seed(1056)
svmFit10CV <- train(Class ~ .,
                    data = GermanCreditTrain,
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    tuneGrid = svmTuneGrid,
                    trControl = trainControl(method = "cv", number = 10)) # {e1071} need to be installed first!
svmFit10CV

set.seed(1056)
system.time(svmFitLOO <- train(Class ~ .,
                   data = GermanCreditTrain,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneGrid = svmTuneGrid,
                   trControl = trainControl(method = "LOOCV")))
# user   system  elapsed 
# 2632.890    2.906  260.825

svmFitLOO

set.seed(1056)
system.time(svmFitLGO <- train(Class ~ .,
                   data = GermanCreditTrain,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneGrid = svmTuneGrid,
                   trControl = trainControl(method = "LGOCV", 
                                            number = 50, 
                                            p = .8)))
svmFitLGO 

set.seed(1056)
system.time(svmFitBoot <- train(Class ~ .,
                    data = GermanCreditTrain,
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    tuneGrid = svmTuneGrid,
                    trControl = trainControl(method = "boot", number = 50)))
# user  system elapsed 
# 171.271   2.006  16.762

svmFitBoot

set.seed(1056)
system.time(svmFitBoot632 <- train(Class ~ .,
                       data = GermanCreditTrain,
                       method = "svmRadial",
                       preProc = c("center", "scale"),
                       tuneGrid = svmTuneGrid,
                       trControl = trainControl(method = "boot632", 
                                                number = 50)))
# user  system elapsed 
# 177.453   2.050  17.174

svmFitBoot632

### Section 4.7 Data Splitting Recommendations (No code!) ####

### Section 4.8 Choosing Between Models ####
set.seed(1056)
system.time(glmProfile <- train(Class ~ .,
                    data = GermanCreditTrain,
                    method = "glm",
                    trControl = trainControl(method = "repeatedcv", 
                                             repeats = 5))) # logistic regression model building by package {glm}.
# user  system elapsed 
# 19.093   0.735   1.300

glmProfile

resamp <- resamples(list(SVM = svmFit, Logistic = glmProfile))
summary(resamp)

## These results are slightly different from those shown in the text.
## There are some differences in the train() function since the 
## original results were produced. This is due to a difference in
## predictions from the ksvm() function when class probs are requested
## and when they are not. See, for example, 
## https://stat.ethz.ch/pipermail/r-help/2013-November/363188.html

modelDifferences <- diff(resamp)
summary(modelDifferences)

## The actual paired t-test:
modelDifferences$statistics$Accuracy

### Section 4.9 Computing ####
## CARET. Relationship between data splitting and trainControl
## https://stackoverflow.com/questions/14968874/caret-relationship-between-data-splitting-and-traincontrol
data(BloodBrain)
?BloodBrain
set.seed(1)
tmp <- createDataPartition(logBBB,p = .8, times = 100)
trControl = trainControl(method = "LGOCV", index = tmp)
ctreeFit <- train(bbbDescr, logBBB, "ctree",trControl=trControl)


methods <- c('boot', 'boot632', 'cv', 
             'repeatedcv', 'LOOCV', 'LGOCV')
n <- 100
tmp <- createDataPartition(logBBB, p = .8, times = n)
208 * 0.8

ll <- lapply(methods, function(x)
  trControl = trainControl(method = x, index = tmp))
lls <- sapply(ll,'[<-','index', NULL)
str(lls)

## Data Splitting
library(AppliedPredictiveModeling)
data(twoClassData) # It has one data frame for features and one vector for class label

dat <- cbind(predictors, classes)

hist(dat$PredictorA)

str(predictors) # check the structure of feature matrix
str(classes) # check the structure of class labels vector

# Set the random number seed so we can reproduce the results
set.seed(1)
# By default, the numbers are returned as a list. Using
# list = FALSE, a matrix of row numbers is generated.
# These samples are allocated to the training set.
?createDataPartition
trainingRows <- createDataPartition(classes,
                                    p = .80,
                                    list= FALSE) # 208*0.8 < 167



head(trainingRows)
# Subset the data into objects for training using
# integer sub-setting.
trainPredictors <- predictors[trainingRows, ]
trainClasses <- classes[trainingRows]
# Do the same for the test set using negative integers.
testPredictors <- predictors[-trainingRows, ]
testClasses <- classes[-trainingRows]
str(trainPredictors)
str(testPredictors)

## Resampling
set.seed(1)
# For illustration, generate the information needed for three
# resampled versions of the training set.
repeatedSplits <- createDataPartition(trainClasses, p = .80, times = 3) # A series of ***test/training*** partitions are created using createDataPartition, attention to other functions in Data Splitting
str(repeatedSplits)

# Similarly, the caret package has functions createResamples (for bootstrapping), createFolds (for k-old cross-validation) and createMultiFolds (for repeated cross-validation). To create indicators for 10-fold cross-validation,
set.seed(1)
cvSplits <- createFolds(trainClasses, k = 10, returnTrain = TRUE)
str(cvSplits) # 151 + 16 or 150 + 17

unique(cvSplits[[1]]) # 151
#diff(1:167, unique(cvSplits[[1]]))

# Get the first set of row numbers from the list.
fold1 <- cvSplits[[1]]
cvPredictors1 <- trainPredictors[fold1,]
cvClasses1 <- trainClasses[fold1]
nrow(trainPredictors)
nrow(cvPredictors1)

## Basic Model Building in R
# modelFunction(price ~ numBedrooms + numBaths + acres, data = housingData)
# modelFunction(x = housePredictors, y = price)


trainPredictors <- as.matrix(trainPredictors)
knnFit <- knn3(x = trainPredictors, y = trainClasses, k = 5)
knnFit

testPredictions <- predict(knnFit, newdata = testPredictors, type = "class")
head(testPredictions)
str(testPredictions)

## Determination of Tuning Parameters
library(caret)
data(GermanCredit)

# set.seed(1056)
# svmFit <- train(Class ~ ., data = GermanCreditTrain,
                # The "method" argument indicates the model type.
                # See ?train for a list of available models.
                method = "svmRadial")

# set.seed(1056)
# svmFit <- train(Class ~ ., data = GermanCreditTrain, method = "svmRadial", preProc = c("center", "scale"))

set.seed(1056)
svmFit <- train(Class ~ ., data = GermanCreditTrain, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 10)


# set.seed(1056)
# svmFit <- train(Class ~ .,data = GermanCreditTrain,method = "svmRadial",preProc = c("center", "scale"),tuneLength = 10,trControl = trainControl(method = "repeatedcv", repeats = 5))
svmFit


# A line plot of the average performance
plot(svmFit, scales = list(x = list(log = 2)))

predictedClasses <- predict(svmFit, GermanCreditTest)
str(predictedClasses)

# Use the "type" option to get class probabilities
predictedProbs <- predict(svmFit, newdata = GermanCreditTest, type = "prob")
head(predictedProbs)

## Between-Model Comparisons
set.seed(1056)
# system.time(logisticReg <- train(Class ~ ., data = GermanCreditTrain, method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 5)))
# user  system elapsed 
# 16.776   1.100   1.322
system.time(logisticReg <- train(Class ~ ., data = GermanCreditTrain, method = "glm", preProc = c("center", "scale")))
logisticReg

resamp <- resamples(list(SVM = svmFit, Logistic = logisticReg))
summary(resamp)


modelDifferences <- diff(resamp)
summary(modelDifferences)


################################################################################
### Session Information

# save(svmFit, svmFit10CV, svmFitLOO, svmFitLGO, svmFitBoot, svmFitBoot632, glmProfile, logisticReg, file = "all_04.RData")

sessionInfo()

q("no")



