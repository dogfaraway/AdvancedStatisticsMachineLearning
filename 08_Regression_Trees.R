################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com) 
###
### Chapter 8: Regression Trees and Rule-Based Models 
###
### Required packages: AppliedPredictiveModeling, caret, Cubis, doMC (optional),
###                    gbm, lattice, party, partykit, randomForest, rpart, RWeka
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

### Create a control function that will be used across models. We
### create the fold assignments explicitly instead of relying on the
### random number seed being set to identical values.

library(caret)
set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)
?createFolds

################################################################################
### Section 8.1 Basic Regression Trees ####

### Demo of minimizing sum of squared errors after partitioning data into two groups
x <- runif(100, 10, 100)
x

x - mean(x)
(x - mean(x))^2
sum((x - mean(x))^2)

x1 <- x[1:50]
x2 <- x[51:100]
sum((x1 - mean(x1))^2) + sum((x2 - mean(x2))^2)


x1 <- x[1:30]
x2 <- x[31:100]
sum((x1 - mean(x1))^2) + sum((x2 - mean(x2))^2)
###

library(rpart)

### Fit two CART models to show the initial splitting process. rpart 
### only uses formulas, so we put the predictors and outcome into
### a common data frame first.

trainData <- solTrainXtrans
trainData$y <- solTrainY

rpStump <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 1))
library(rpart.plot)
rpart.plot(rpStump)

rpSmall <- rpart(y ~ ., data = trainData, 
                 control = rpart.control(maxdepth = 2))
rpart.plot(rpStump)

### Tune the model
library(doMC)
?registerDoMC
options("cores")
registerDoMC(4)

library(caret)

set.seed(100)
system.time(cartTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "rpart",
                  tuneLength = 25,
                  trControl = ctrl))
# user  system elapsed 
# 3.581   0.443   3.051
cartTune
## cartTune$finalModel


### Plot the tuning results
plot(cartTune, scales = list(x = list(log = 10)))

### Use the partykit package to make some nice plots. First, convert
### the rpart objects to party objects.

# library(partykit)
# 
# cartTree <- as.party(cartTune$finalModel)
# plot(cartTree)

### Get the variable importance. 'competes' is an argument that
### controls whether splits not used in the tree should be included
### in the importance calculations.

cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
cartImp # only 20 most important variables shown (out of 228)

### Save the test set results in a data frame                 
testResults <- data.frame(obs = solTestY,
                          CART = predict(cartTune, solTestXtrans))

### Tune the conditional inference tree

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2)))) # why not cGrid <- data.frame(mincriterion = c(.75, .95, .99))

set.seed(100)
system.time(ctreeTune <- train(x = solTrainXtrans, y = solTrainY,
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl))
# user  system elapsed 
# 16.082   2.482   6.973
ctreeTune
plot(ctreeTune)

##ctreeTune$finalModel               
plot(ctreeTune$finalModel)

testResults$cTree <- predict(ctreeTune, solTestXtrans)

################################################################################
### Section 8.2 Regression Model Trees and 8.3 Rule-Based Models ####

### Tune the model tree. Using method = "M5" actually tunes over the
### tree- and rule-based versions of the model. M = 10 is also passed
### in to make sure that there are larger terminal nodes for the
### regression models.

set.seed(100)
system.time(m5Tune <- train(x = solTrainXtrans, y = solTrainY,
                method = "M5",
                trControl = ctrl,
                control = Weka_control(M = 10)))
m5Tune

plot(m5Tune)

## m5Tune$finalModel

## plot(m5Tune$finalModel)

### Show the rule-based model too

ruleFit <- M5Rules(y~., data = trainData, control = Weka_control(M = 10))
ruleFit

################################################################################
### Section 8.4 Bagged Trees ####

### Optional: parallel processing can be used via the 'do' packages,
### such as doMC, doMPI etc. We used doMC (not on Windows) to speed
### up the computations.

### WARNING: Be aware of how much memory is needed to parallel
### process. It can very quickly overwhelm the available hardware. The
### estimate of the median memory usage (VSIZE = total memory size) 
### was 9706M for a core, but could range up to 9706M. This becomes 
### severe when parallelizing randomForest() and (especially) calls 
### to cforest(). 

### WARNING 2: The RWeka package does not work well with some forms of
### parallel processing, such as mutlicore (i.e. doMC).

library(doMC)
registerDoMC(5)

set.seed(100)

system.time(treebagTune <- train(x = solTrainXtrans, y = solTrainY,
                     method = "treebag",
                     nbagg = 50,
                     trControl = ctrl))

treebagTune

################################################################################
### Section 8.5 Random Forests ####

mtryGrid <- data.frame(mtry = floor(seq(10, ncol(solTrainXtrans), length = 10)))


### Tune the model using cross-validation
set.seed(100)
system.time(rfTune <- train(x = solTrainXtrans, y = solTrainY,
                method = "rf",
                tuneGrid = mtryGrid,
                ntree = 1000,
                importance = TRUE,
                trControl = ctrl))
rfTune

plot(rfTune)

rfImp <- varImp(rfTune, scale = FALSE)
rfImp

### Tune the model using the OOB estimates
ctrlOOB <- trainControl(method = "oob")
set.seed(100)
system.time(rfTuneOOB <- train(x = solTrainXtrans, y = solTrainY,
                   method = "rf",
                   tuneGrid = mtryGrid,
                   ntree = 1000,
                   importance = TRUE,
                   trControl = ctrlOOB))
rfTuneOOB

plot(rfTuneOOB)

### Predictions on the Test Set ??? ####

### Tune the conditional inference forests ####
set.seed(100)
system.time(condrfTune <- train(x = solTrainXtrans, y = solTrainY,
                    method = "cforest",
                    tuneGrid = mtryGrid,
                    controls = cforest_unbiased(ntree = 1000),
                    trControl = ctrl))
condrfTune

plot(condrfTune)

set.seed(100)
system.time(condrfTuneOOB <- train(x = solTrainXtrans, y = solTrainY,
                       method = "cforest",
                       tuneGrid = mtryGrid,
                       controls = cforest_unbiased(ntree = 1000),
                       trControl = trainControl(method = "oob")))
condrfTuneOOB

plot(condrfTuneOOB)

################################################################################
### Section 8.6 Boosting ####

gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(100, 1000, by = 50),
                       shrinkage = c(0.01, 0.1))
set.seed(100)
system.time(gbmTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl,
                 verbose = FALSE))
gbmTune

plot(gbmTune, auto.key = list(columns = 4, lines = TRUE))

gbmImp <- varImp(gbmTune, scale = FALSE)
gbmImp

################################################################################
### Section 8.7 Cubist ####

cbGrid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(100)
system.time(cubistTune <- train(solTrainXtrans, solTrainY,
                    "cubist",
                    tuneGrid = cbGrid,
                    trControl = ctrl))
cubistTune
save(cartTune, ctreeTune, m5Tune, cubistTune, treebagTune, rfTune, rfTuneOOB, condrfTune, condrfTuneOOB, gbmTune, file="all_08.RData")

plot(cubistTune, auto.key = list(columns = 4, lines = TRUE))

cbImp <- varImp(cubistTune, scale = FALSE)
cbImp

################################################################################
### Session Information

sessionInfo()

q("no")

### Section 8.8 Computing ####
# The R packages used in this section are caret, Cubist, gbm, ipred, party, partykit,
# randomForest, rpart, RWeka.
# Single Trees
# Two widely used implementations for single regression trees in R are rpart and
# party. The rpart package makes splits based on the CART methodology using
# the rpart function, whereas the party makes splits based on the conditional
# inference framework using the ctree function. Both rpart and ctree functions
# use the formula method:

library(rpart) # Recursive PARTition
data(iris)
irisRpart <- rpart(Species ~ ., data = iris)
library(rpart.plot)
class(irisRpart) # "rpart"
rpart.plot(irisRpart) # plot.rpart()
#text(irisRpart)
irisRpart

library(tree)
irisTree <- tree(Species ~ ., data = iris)
plot(irisTree)
text(irisTree)
?tree
irisTreePr <- snip.tree(irisTree, c(7,12))
plot(irisTreePr)
text(irisTreePr)

?airquality
data(airquality)
solarRpart <- rpart(Solar.R ~  Ozone+Wind+Temp+Month+Day, data = airquality)
rpart.plot(solarRpart) # plot.rpart()
#text(solarRpart)

library(partykit)
solarRpart2 <- as.party(solarRpart)
plot(solarRpart2)

rpartTree <- rpart(y ~ ., data = trainData)
# or,
ctreeTree <- ctree(y ~ ., data = trainData)
# The rpart function has several control parameters that can be accessed
# through the rpart.control argument. Two that are commonly used in training
# and that can be accessed through the train function are the complexity
# parameter (cp) and maximum node depth (maxdepth). To tune an CART
# tree over the complexity parameter, the method option in the train function
# should be set to method = "rpart". To tune overmaximumdepth, themethod
# option should be set to method="rpart2":
set.seed(100)
rpartTune <- train(solTrainXtrans, solTrainY,
                   + method = "rpart2",
                   + tuneLength = 10,
                   + trControl = trainControl(method = "cv"))

# Likewise, the party package has several control parameters that can be
# accessed through the ctree_control argument. Two of these parameters are
# commonly used in training: mincriterion and maxdepth. mincriterion defines
# the statistical criterion that must be met in order to continue splitting;
# maxdepth is the maximum depth of the tree. To tune a conditional inference
# tree over mincriterion, the method option in the train function should be
# set to method = "ctree". To tune over maximum depth, the method option
# should be set to method="ctree2".
# The plot method in the party package can produce the tree diagrams shown
# in Fig. 8.4 via
plot(treeObject)

# To produce such plots for rpart trees, the partykit can be used to first convert
# the rpart object to a party object and then use the plot function:
library(partykit)
rpartTree2 <- as.party(rpartTree)
plot(rpartTree2)

# Model Trees
# The main implementation for model trees can be found in the Weka software
# suite, but the model can be accessed in R using the RWeka package. There
# are two different interfaces: M5P fits the model tree, while M5Rules uses the rulebased
# version. In either case, the functions work with formula methods:
library(RWeka)
m5tree <- M5P(y ~ ., data = trainData)
# or, for rules:
# m5rules <- M5Rules(y ~ ., data = trainData)
# In our example, the minimum number of training set points required to
# create additional splits was raised from the default of 4–10. To do this, the
# control argument is used:
m5tree <- M5P(y ~ ., data = trainData,
              + control = Weka_control(M = 10))
# The control argument also has options for toggling the use of smoothing and
# pruning. If the full model tree is used, a visualization similar to Fig. 8.10 can
# be created by the plot function on the output from M5P.
# To tune these models, the train function in the caret package has two
# options: using method = "M5" evaluates model trees and the rule-based versions
# of the model, as well as the use of smoothing and pruning. Figure 8.12 shows
# the results of evaluating these models from the code:
library(caret)
library(AppliedPredictiveModeling)
data(solubility)
set.seed(100)
system.time(m5Tune <- train(solTrainXtrans, solTrainY,
                            method = "M5",
                            trControl = trainControl(method = "cv"),
                            ## Use an option for M5() to specify the minimum
                            ## number of samples needed to further splits the
                            ## data to be 10
                            control = Weka_control(M = 10)))
# followed by plot(m5Tune). train with method = "M5Rules" evaluates only the
# rule-based version of the model.
plot(m5Tune)
m5Tune

# Bagged Trees
# The ipred package contains two functions for bagged trees: bagging uses the
# formula interface and ipredbagg has the non-formula interface:
library(ipred) # Improved PREDictors
system.time(baggedTree <- ipredbagg(solTrainY, solTrainXtrans))
names(baggedTree)
baggedTree$mtrees # 25 trees inside

rpart.plot(baggedTree$mtrees[[1]]$btree)
rpart.plot(baggedTree$mtrees[[2]]$btree)

## or
baggedTree <- bagging(y ~ ., data = trainData)
# The function uses the rpart function and details about the type of tree can
# be specified by passing rpart.control to the control argument for bagging and
# ipredbagg. By default, the largest possible tree is created.
# Several other packages have functions for bagging. The aforementioned
# RWeka package has a function called Bagging and the caret package has a
# general framework for bagging many model types, including trees, called bag.
# Conditional inference trees can also be bagged using the cforest function in
# the party package if the argument mtry is equal to the number of predictors:
library(party)
## The mtry parameter should be the number of predictors (the
## number of columns minus 1 for the outcome).
bagCtrl <- cforest_control(mtry = ncol(trainData) - 1)
baggedTree <- cforest(y ~ ., data = trainData, controls = bagCtrl)

# Random Forest
# The primary implementation for random forest comes from the package with
# the same name:
library(randomForest)
rfModel <- randomForest(solTrainXtrans, solTrainY)
## or
rfModel <- randomForest(y ~ ., data = trainData)

# The two main arguments are mtry for the number of predictors that are
# randomly sampled as candidates for each split and ntree for the number of
# bootstrap samples. The default for mtry in regression is the number of predictors
# divided by 3. The number of trees should be large enough to provide a
# stable, reproducible results. Although the default is 500, at least 1,000 bootstrap
# samples should be used (and perhaps more depending on the number of
#                         predictors and the values of mtry). Another important option is importance;
# by default, variable importance scores are not computed as they are time
# consuming; importance = TRUE will generate these values:
library(randomForest)
rfModel <- randomForest(solTrainXtrans, solTrainY,
                        + importance = TRUE,
                        + ntrees = 1000)
# For forests built using conditional inference trees, the cforest function in
# the party package is available. It has similar options, but the controls argument
# (note the plural) allows the user to pick the type of splitting algorithm
# to use (e.g., biased or unbiased).
# Neither of these functions can be used with missing data.
# The train function contains wrappers for tuning either of these models by
# specifying either method = "rf" or method = "cforest". Optimizing the mtry
# parameter may result in a slight increase in performance. Also, train can use
# standard resampling methods for estimating performance (as opposed to the
#                                                         out-of-bag estimate).
# For randomForest models, the variable importance scores can be accessed
# using a function in that package called importance. For cforest objects, the
# analogous function in the party package is varimp.
# Each package tends to have its own function for calculating importance
# scores, similar to the situation for class probabilities shown in Table B.1 of the
# first Appendix. caret has a unifying function called varImp that is a wrapper
# for variable importance functions for the following tree-model objects: rpart,
# classbagg (produced by the ipred package’s bagging functions) randomForest,
# cforest, gbm, and cubist.

# Boosted Trees
# The most widely used package for boosting regression trees via stochastic
# gradient boosting machines is gbm. Like the randomforests interface,models
# can be built in two distinct ways:
library(gbm)
gbmModel <- gbm.fit(solTrainXtrans, solTrainY, distribution = "gaussian")
## or
gbmModel <- gbm(y ~ ., data = trainData, distribution = "gaussian")

# The distribution argument defines the type of loss function that will be
# optimized during boosting. For a continuous response, distribution should
# be set to “gaussian.”The number of trees (n.trees), depth of trees (interaction.
#                                                                     depth), shrinkage (shrinkage), and proportion of observations to be sampled
# (bag.fraction) can all be directly set in the call to gbm.
# Like other parameters, the train function can be used to tune over these
# parameters. To tune over interaction depth, number of trees, and shrinkage,
# for example, we first define a tuning grid. Then we train over this grid as
# follows:
gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       + .n.trees = seq(100, 1000, by = 50),
                       + .shrinkage = c(0.01, 0.1))
set.seed(100)
gbmTune <- train(solTrainXtrans, solTrainY,
                 + method = "gbm",
                 + tuneGrid = gbmGrid,
                 + ## The gbm() function produces copious amounts
                   + ## of output, so pass in the verbose option
                   + ## to avoid printing a lot to the screen.
                   + verbose = FALSE)
# Cubist
# As previously mentioned, the implementation for this model created by Rule-
#   Quest was recently made public using an open-source license. An R package
# called Cubist was created using the open-source code. The function does not
# have a formula method since it is desirable to have the Cubist code manage
# the creation and usage of dummy variables. To create a simple rule-based
# model with a single committee and no instance-based adjustment, we can
# use the simple code:
library(Cubist)
# cubistMod <- cubist(solTrainXtrans, solTrainY)
# An argument, committees, fits multiple models. The familiar predict method
# would be used for new samples:
predict(cubistMod, solTestXtrans)
# The choice of instance-based corrections does not need to be made until
# samples are predicted. The predict function has an argument, neighbors, that
# can take on a single integer value (between 0 and 9) to adjust the rule-based
# predictions from the training set.
# Once the model is trained, the summary function generates the exact rules
# that were used, as well as the final smoothed linear model for each rule. Also,as with most other models, the train function in the caret package can tune
# the model over values of committees and neighbors through resampling:
cubistTuned <- train(solTrainXtrans, solTrainY, method = "cubist")

Exercises
#8.1. Recreate the simulated data from Exercise 7.2:
library(mlbench)
set.seed(200)
simulated <- mlbench.friedman1(200, sd = 1)
simulated <- cbind(simulated$x, simulated$y)
simulated <- as.data.frame(simulated)
colnames(simulated)[ncol(simulated)] <- "y"
# (a) Fit a random forest model to all of the predictors, then estimate the
# variable importance scores:
library(randomForest)
library(caret)
model1 <- randomForest(y ~ ., data = simulated,
                       + importance = TRUE,
                       + ntree = 1000)
rfImp1 <- varImp(model1, scale = FALSE)
# Did the random forest model significantly use the uninformative predictors
# (V6 – V10)?
# (b) Now add an additional predictor that is highly correlated with one of the
# informative predictors. For example:
simulated$duplicate1 <- simulated$V1 + rnorm(200) * .1
cor(simulated$duplicate1, simulated$V1)
# Fit another random forest model to these data. Did the importance score
# for V1 change? What happens when you add another predictor that is
# also highly correlated with V1?
# (c) Use the cforest function in the party package to fit a random forest model
# using conditional inference trees. The party package function varimp can
# calculate predictor importance. The conditional argument of that function
# toggles between the traditional importance measure and the modified
# version described in Strobl et al. (2007). Do these importances show the
# same pattern as the traditional random forest model?
# (d) Repeat this process with different tree models, such as boosted trees and
# Cubist. Does the same pattern occur?

# 8.2. Use a simulation to show tree bias with different granularities.
# 8.3. In stochastic gradient boosting the bagging fraction and learning rate
# will govern the construction of the trees as they are guided by the gradient.
# Although the optimal values of these parameters should be obtained
# through the tuning process, it is helpful to understand how the magnitudes
# of these parameters affect magnitudes of variable importance. Figure 8.24
# provides the variable importance plots for boosting using two extreme values
# for the bagging fraction (0.1 and 0.9) and the learning rate (0.1 and 0.9) for
# the solubility data. The left-hand plot has both parameters set to 0.1, and
# the right-hand plot has both set to 0.9:
#   (a) Why does the model on the right focus its importance on just the first few
# of predictors, whereas the model on the left spreads importance across
# more predictors?
# (b) Which model do you think would be more predictive of other samples?
# (c) How would increasing interaction depth affect the slope of predictor importance
# for either model in Fig. 8.24?
# 8.2. Use a simulation to show tree bias with different granularities.
# 8.3. In stochastic gradient boosting the bagging fraction and learning rate
# will govern the construction of the trees as they are guided by the gradient.
# Although the optimal values of these parameters should be obtained
# through the tuning process, it is helpful to understand how the magnitudes
# of these parameters affect magnitudes of variable importance. Figure 8.24
# provides the variable importance plots for boosting using two extreme values
# for the bagging fraction (0.1 and 0.9) and the learning rate (0.1 and 0.9) for
# the solubility data. The left-hand plot has both parameters set to 0.1, and
# the right-hand plot has both set to 0.9:
#   (a) Why does the model on the right focus its importance on just the first few
# of predictors, whereas the model on the left spreads importance across
# more predictors?
# (b) Which model do you think would be more predictive of other samples?
# (c) How would increasing interaction depth affect the slope of predictor importance
# for either model in Fig. 8.24?



