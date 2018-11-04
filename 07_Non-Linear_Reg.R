################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 7: Non-Linear Regression Models
###
### Required packages: AppliedPredictiveModeling, caret, doMC (optional), earth,
###                    kernlab, lattice, nnet
###
### Data used: The solubility from the AppliedPredictiveModeling package
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

################################################################################
### Load the data

library(AppliedPredictiveModeling)
data(solubility)

### Create a control funciton that will be used across models. We
### create the fold assignments explicitly instead of relying on the
### random number seed being set to identical values.

library(caret)
set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

################################################################################
### Section 7.1 Neural Networks ####

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.

### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the availible hardware. We
### estimate the memory usuage (VSIZE = total memory size) to be 
### 2677M/core.

library(doMC)
?registerDoMC
options("cores")
registerDoMC(12)


library(caret)

nnetGrid <- expand.grid(decay = c(0.00, 0.01, 0.10), size = c(1, 3, 5, 7, 9, 11, 13), bag = FALSE)

set.seed(100)
system.time(nnetTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 13 * (ncol(solTrainXtrans) + 1) + 13 + 1,
                  maxit = 1000,
                  allowParallel = TRUE))
# 117578.094 72.592 10151.221
?train
nnetTune
#save(nnetTune, file="nnetTune.RData")

plot(nnetTune)

testResults <- data.frame(obs = solTestY,
                          NNet = predict(nnetTune, solTestXtrans))

################################################################################
### Section 7.2 Multivariate Adaptive Regression Splines ####

set.seed(100)
system.time(marsTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "earth",
                  tuneGrid = expand.grid(degree = 1, nprune = 2:38),
                  trControl = ctrl))
#    user  system elapsed 
# 368.153   1.729  59.245 

marsTune
#save(marsTune, file="marsTune.RData")

plot(marsTune)

testResults$MARS <- predict(marsTune, solTestXtrans)

marsImp <- varImp(marsTune, scale = FALSE)
plot(marsImp, top = 25)

################################################################################
### Section 7.3 Support Vector Machines ####

## In a recent update to caret, the method to estimate the
## sigma parameter was slightly changed. These results will
## slightly differ from the text for that reason.

set.seed(100)
system.time(svmRTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength = 14,
                  trControl = ctrl))


svmRTune
#save(svmRTune, file="svmRTune.RData")

plot(svmRTune, scales = list(x = list(log = 2)))                 

svmGrid <- expand.grid(degree = 1:2, 
                       scale = c(0.01, 0.005, 0.001), 
                       C = 2^(-2:5))
set.seed(100)
system.time(svmPTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "svmPoly",
                  preProc = c("center", "scale"),
                  tuneGrid = svmGrid,
                  trControl = ctrl))
# 1083.976 1.745 183.900

svmPTune
#save(svmPTune, file="svmPTune.RData")

plot(svmPTune, 
     scales = list(x = list(log = 2), 
                   between = list(x = .5, y = 1)))                 

testResults$SVMr <- predict(svmRTune, solTestXtrans)
testResults$SVMp <- predict(svmPTune, solTestXtrans)

################################################################################
### Section 7.4 K-Nearest Neighbors ####

### First we remove near-zero variance predictors
knnDescr <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)]

set.seed(100)
system.time(knnTune <- train(x = knnDescr, y = solTrainY,
                 method = "knn",
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(k = 1:20),
                 trControl = ctrl))
# 318.006 1.538 14.289

knnTune
#save(knnTune, file="knnTune.RData")
save(knnTune, marsTune, nnetTune, svmPTune, svmRTune, file="all_07.RData")

plot(knnTune)

testResults$Knn <- predict(svmRTune, solTestXtrans[, names(knnDescr)])

################################################################################
### Session Information

sessionInfo()

q("no")

### Section 7.5 Computing ####
# Neural Networks
?nnet
nnetFit <- nnet(predictors, outcome,
 size = 5,
                 decay = 0.01,
                 linout = TRUE,
                 ## Reduce the amount of printed output
                 trace = FALSE,
                 ## Expand the number of iterations to find
                 ## parameter estimates..
 maxit = 500,
 ## and the number of parameters used by the model
 MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)

library(AppliedPredictiveModeling)
data(solubility)

## The findCorrelation takes a correlation matrix and determines the > ## column numbers that should be removed to keep all pair-wise
## correlations below a threshold
tooHigh <- findCorrelation(cor(solTrainXtrans), cutoff = .75)
trainXnnet <- solTrainXtrans[, -tooHigh]
testXnnet <- solTestXtrans[, -tooHigh]
## Create a specific candidate set of models to evaluate:
nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                         .size = c(1:10),
                         ## The next option is to use bagging (see the
                         ## next chapter) instead of different random
 ## seeds.
                         .bag = FALSE)
set.seed(100)
ctrl <- trainControl(method = "cv", number = 10)
system.time(nnetTune <- train(solTrainXtrans, solTrainY,
                   method = "avNNet",
                   tuneGrid = nnetGrid,
                   trControl = ctrl,
 ## Automatically standardize data prior to modeling
                   ## and prediction
                   preProc = c("center", "scale"),
                   linout = TRUE,
                   trace = FALSE,
 MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
 maxit = 500))

# Multivariate Adaptive Regression Splines
marsFit <- earth(solTrainXtrans, solTrainY)
marsFit
summary(marsFit)

# Define the candidate models to test
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
# Fix the seed so that the results can be reproduced
set.seed(100)
marsTuned <- train(solTrainXtrans, solTrainY,
                    method = "earth",
 # Explicitly declare the candidate models to test
                    tuneGrid = marsGrid,
                    trControl = trainControl(method = "cv"))

head(predict(marsTuned, solTestXtrans))
arImp(marsTuned)

# Support Vector Machines
library(kernlab)
svmFit <- ksvm(x = solTrainXtrans, y = solTrainY,
                kernel ="rbfdot", kpar = "automatic",
                C = 1, epsilon = 0.1)

svmRTuned <- train(solTrainXtrans, solTrainY,
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    tuneLength = 14,
                    trControl = trainControl(method = "cv"))
svmRTuned

svmRTuned$finalModel
