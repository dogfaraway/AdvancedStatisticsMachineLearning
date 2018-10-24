################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 6: Linear Regression and Its Cousins
###
### Required packages: AppliedPredictiveModeling, lattice, corrplot, pls, 
###                    elasticnet,
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

### Section 6.1 Case Study: Quantitative Structure- Activity Relationship Modeling ####

library(AppliedPredictiveModeling)
data(solubility)
help(solTrainX)
str(solTrainX)
names(solTrainX) # 1~208: Two hundred and eight binary “fingerprints” that indicate the presence or absence of a particular chemical substructure, 210~225: Sixteen count descriptors, such as the number of bonds or the number of bromine atoms, 209, 226~228: Four continuous descriptors, such as molecular weight or surface area.

summary(solTrainY) # The outcome data were measured on the log10 scale and ranged from −11.6 to 1.6 with an average log solubility value of −2.7.
# summary(solTrainX)

corTRX <- cor(solTrainX) # 47 pairs have correlations greater than 0.90.
str(corTRX)
diag(corTRX) <- 0
corTRX[lower.tri(corTRX)] <- 0
which(abs(cor(solTrainX)) > 0.9, arr.ind=TRUE)
unique(which(abs(cor(solTrainX)) > 0.9, arr.ind=TRUE)[,1])

library(lattice)

### Some initial plots of the data

xyplot(solTrainY ~ solTrainX$MolWeight, type = c("p", "g"), # griding
       ylab = "Solubility (log)",
       main = "(a)",
       xlab = "Molecular Weight")
xyplot(solTrainY ~ solTrainX$NumRotBonds, type = c("p", "g"),
       ylab = "Solubility (log)",
       xlab = "Number of Rotatable Bonds")

names(solTrainX)[100] # "FP100"
bwplot(solTrainY ~ ifelse(solTrainX[,100] == 1, "structure present", "structure absent"), # it's a character vector of length 951 with two possible values, so smart !
       ylab = "Solubility (log)",
       main = "(b)",
       horizontal = FALSE)

### Find the columns that are not fingerprints (i.e. the continuous
### predictors). grep will return a list of integers corresponding to
### column names that contain the pattern "FP".

Fingerprints <- grep("FP", names(solTrainXtrans)) # Since there are only two values of these binary variables, there is very little that pre-processing will accomplish.

library(e1071)
summary(sapply(solTrainX[,-Fingerprints], skewness))

library(caret)
featurePlot(solTrainXtrans[, -Fingerprints],
            solTrainY,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"),
            labels = rep("", 2))

library(corrplot)

### We used the full namespace to call this function because the pls
### package (also used in this chapter) has a function with the same
### name.

corrplot::corrplot(cor(solTrainXtrans[, -Fingerprints]), order = "hclust", tl.cex = .8)

corrplot::corrplot(cor(solTrainXtrans[, -Fingerprints]), order = "hclust", tl.cex = .8, method = c("number"))

################################################################################
### Section 6.2 Linear Regression ####

### Create a control function that will be used across models. We
### create the fold assignments explicitly instead of relying on the
### random number seed being set to identical values.

set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE) # y: a vector of outcomes; returnTrain: a logical. When true, the values returned are the sample positions corresponding to the data used during training. This argument only works in conjunction with list = TRUE
ctrl <- trainControl(method = "cv", index = indx) # index: a list with elements for each resampling iteration. Each list element is a vector of integers corresponding to the rows used for training at that iteration.
?trainControl

### Linear regression model with all of the predictors. This will
### produce some warnings that a 'rank-deficient fit may be
### misleading'. This is related to the predictors being so highly
### correlated that some of the math has broken down.

set.seed(100)
lmTune0 <- train(x = solTrainXtrans, y = solTrainY,
                 method = "lm",
                 trControl = ctrl)

lmTune0

dat <- cbind(solTrainXtrans, solTrainY) # for model formula syntax
?lm
names(dat)
summary(lm(solTrainY ~ ., data=dat))

### And another using a set of predictors reduced by unsupervised
### filtering. We apply a filter to reduce extreme between-predictor
### correlations. Note the lack of warnings.

tooHigh <- findCorrelation(cor(solTrainXtrans), .9)
trainXfiltered <- solTrainXtrans[, -tooHigh]
testXfiltered  <-  solTestXtrans[, -tooHigh]

set.seed(100)
lmTune <- train(x = trainXfiltered, y = solTrainY,
                method = "lm",
                trControl = ctrl)

lmTune

### Save the test set results in a data frame                 
testResults <- data.frame(obs = solTestY,
                          Linear_Regression = predict(lmTune, testXfiltered))


################################################################################
### Section 6.3 Partial Least Squares ####

## Run PLS and PCR on solubility data and compare results
set.seed(100)
plsTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "pls",
                 tuneGrid = expand.grid(ncomp = 1:20),
                 trControl = ctrl)
plsTune

testResults$PLS <- predict(plsTune, solTestXtrans)

set.seed(100)
pcrTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "pcr",
                 tuneGrid = expand.grid(ncomp = 1:35),
                 trControl = ctrl)
pcrTune                  

plsResamples <- plsTune$results
plsResamples$Model <- "PLS"
pcrResamples <- pcrTune$results
pcrResamples$Model <- "PCR"
plsPlotData <- rbind(plsResamples, pcrResamples)

xyplot(RMSE ~ ncomp,
       data = plsPlotData,
       #aspect = 1,
       xlab = "# Components",
       ylab = "RMSE (Cross-Validation)",
       auto.key = list(columns = 2),
       groups = Model,
       type = c("o", "g"))

plsImp <- varImp(plsTune, scale = FALSE)
plot(plsImp, top = 25, scales = list(y = list(cex = .95)))

################################################################################
### Section 6.4 Penalized Models ####

## The text used the elasticnet to obtain a ridge regression model.
## There is now a simple ridge regression method.

ridgeGrid <- expand.grid(lambda = seq(0, .1, length = 15))

set.seed(100)
ridgeTune <- train(x = solTrainXtrans, y = solTrainY,
                   method = "ridge",
                   tuneGrid = ridgeGrid,
                   trControl = ctrl,
                   preProc = c("center", "scale"))
ridgeTune

print(update(plot(ridgeTune), xlab = "Penalty"))


enetGrid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))
enetTune
class(enetTune) # "train"
enetTune$modelInfo$parameters

                                  plot(enetTune)
?plot.train # ggplot.train {caret}
testResults$Enet <- predict(enetTune, solTestXtrans) # and then ...

################################################################################
### Session Information

sessionInfo()

q("no")

### Section 6.5 Computing ####
library(AppliedPredictiveModeling)
data(solubility)
help(trainX)
## The data objects begin with "sol":
ls(pattern = "^solT")
names(solTrainX)

set.seed(2)
sample(names(solTrainX), 8)

trainingData <- solTrainXtrans
## Add the solubility outcome
trainingData$Solubility <- solTrainY
names(trainingData)

lmFitAllPredictors <- lm(Solubility ~ ., data = trainingData)

summary(lmFitAllPredictors)

lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)
head(lmPred1)

lmValues1 <- data.frame(obs = solTestY, pred = lmPred1)
defaultSummary(lmValues1)

rlmFitAllPredictors <- rlm(Solubility ~ ., data = trainingData)

ctrl <- trainControl(method = "cv", number = 10)

set.seed(100)
lmFit1 <- train(x = solTrainXtrans, y = solTrainY,
                method = "lm", trControl = ctrl)

lmFit1

xyplot(solTrainY ~ predict(lmFit1),
       ## plot the points (type = 'p') and a background grid ('g')
      type = c("p", "g"),
      xlab = "Predicted", ylab = "Observed")

xyplot(resid(lmFit1) ~ predict(lmFit1),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")


corThresh <- .9
tooHigh <- findCorrelation(cor(solTrainXtrans), corThresh)
corrPred <- names(solTrainXtrans)[tooHigh]
trainXfiltered <- solTrainXtrans[, -tooHigh]

testXfiltered <- solTestXtrans[, -tooHigh]
set.seed(100)
lmFiltered <- train(solTrainXtrans, solTrainY, method = "lm",
                    trControl = ctrl)
lmFiltered


set.seed(100)
rlmPCA <- train(solTrainXtrans, solTrainY,
                method = "rlm",
                preProcess = "pca",
                trControl = ctrl)
rlmPCA


plsFit <- plsr(Solubility ~ ., data = trainingData)


predict(plsFit, solTestXtrans[1:5,], ncomp = 1:2)


set.seed(100)
plsTune <- train(solTrainXtrans, solTrainY,
                 method = "pls",
                 ## The default tuning grid evaluates
                 ## components 1... tuneLength
                 tuneLength = 20,
                 trControl = ctrl,
                 preProc = c("center", "scale"))



ridgeModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY,
                   lambda = 0.001)

ridgePred <- predict(ridgeModel, newx = as.matrix(solTestXtrans),
                     s= 1, mode = "fraction",+ type = "fit")
head(ridgePred$fit)

#To tune over the penalty, train can be used with a different method:
## Define the candidate set of values
ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
set.seed(100)
ridgeRegFit <- train(solTrainXtrans, solTrainY,
                     method = "ridge",
                     ## Fit the model over many penalty values
                     tuneGrid = ridgeGrid,
                     trControl = ctrl,
                     ## put the predictors on the same scale
                     preProc = c("center", "scale"))
ridgeRegFit

# The lasso model can be estimated using a number of different functions.
# The lars package contains the lars function, the elasticnet package has enet,
# and the glmnet package has a function of the same name. The syntax for
# these functions is very similar. For the enet function, the usage would be

enetModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY,
                  + lambda = 0.01, normalize = TRUE)


enetPred <- predict(enetModel, newx = as.matrix(solTestXtrans),
                    s= .1, mode = "fraction",
                    type = "fit")
## A list is returned with several items:
names(enetPred)

## The 'fit' component has the predicted values:
head(enetPred$fit)

# To determine which predictors are used in the model, the predict method is
# used with type = "coefficients":
enetCoef <- predict(enetModel, newx = as.matrix(solTestXtrans),
                   s= .1, mode = "fraction",
                   type = "coefficients")
tail(enetCoef$coefficients)


# Other packages to fit the lasso model or some alternate version of the
# model are biglars (for large data sets), FLLat (for the fused lasso), grplasso
# (the group lasso), penalized, relaxo (the relaxed lasso), and others. To tune
# the elastic net model using train, we specify method = "enet". Here, we tune
# the model over a custom set of penalties:
enetGrid <- expand.grid(.lambda = c(0, 0.01, .1),
                        .fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune <- train(solTrainXtrans, solTrainY,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))

