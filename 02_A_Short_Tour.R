################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 2: A Short Tour of the Predictive Modeling Process
###
### Required packages: AppliedPredictiveModeling, earth, caret, lattice
###
### Data used: The FuelEconomy data in the AppliedPredictiveModeling package
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
### Section 2.1 Case Study: Predicting Fuel Economy ####

library(AppliedPredictiveModeling)
data(FuelEconomy)
help(cars2010) # ? is an alias of help()

## The http://fueleconomy.gov website, run by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy and the U.S. Environmental Protection Agency, lists different estimates of fuel economy for passenger cars and trucks. For each vehicle, various characteristics are recorded such as the engine displacement (note: it is the swept volume of all the pistons inside the cylinders of a reciprocating engine in a single movement from top dead centre, TDC, to bottom dead centre, BDC. https://en.wikipedia.org/wiki/Engine_displacement) or number of cylinders. Along with these values, laboratory measurements are made for the city and highway miles per gallon (MPG) of the car.

## Predictors extracted from the website include: EngDispl, NumCyl, Transmission, AirAspirationMethod, NumGears, TransLockup, TransCreeperGear, DriveDesc, IntakeValvePerCyl, ExhaustValvesPerCyl, CarlineClassDesc, VarValveTiming and VarValveLift. The outcome used in the book is in column FE and is the unadjusted highway data.

## Format data for plotting against engine displacement

## Sort by engine displacement
head(cars2010)
cars2010 <- cars2010[order(cars2010$EngDispl),]
head(cars2011)
cars2011 <- cars2011[order(cars2011$EngDispl),]

## Combine data into one data frame
cars2010a <- cars2010
cars2010a$Year <- "2010 Model Year" # Append a new column
cars2011a <- cars2011
cars2011a$Year <- "2011 Model Year" # Append a new column

plotData <- rbind(cars2010a, cars2011a)

library(lattice)
?xyplot
xyplot(FE ~ EngDispl|Year, plotData,
       xlab = "Engine Displacement",
       ylab = "Fuel Efficiency (MPG)",
       between = list(x = 1.2))

## Fit a single linear model and conduct 10-fold CV to estimate the error
library(caret)
set.seed(1)
lm1Fit <- train(FE ~ EngDispl, 
                data = cars2010,
                method = "lm", 
                trControl = trainControl(method= "cv"))
lm1Fit
names(lm1Fit)
lm1Fit$control$index
fitted(lm1Fit)
length(fitted(lm1Fit))

lm1Fit$control$index # indices for training samples
1107 - sapply(lm1Fit$control$index, length)

library(lattice)
xyplot(FE ~ EngDispl, data = cars2010, type = c("p", "r"))

lm1 <- data.frame(Observed = cars2010$FE, Predicted = fitted(lm1Fit))

# xyplot(Predicted ~ Observed, data = lm1)
plot(cars2010$FE, fitted(lm1Fit), xlim = c(10, 72), ylim = c(10, 72))
abline(a = 0, b = 1)

## Returned Value from 10-fold CV
names(lm1Fit)
lm1Fit$trainingData
lm1Fit$method
lm1Fit$modelInfo
lm1Fit$modelType # "Regression"
lm1Fit$results
lm1Fit$pred
lm1Fit$bestTune
lm1Fit$modelInfo

## Fit a quadratic model too

## Create squared terms
cars2010$ED2 <- cars2010$EngDispl^2
cars2011$ED2 <- cars2011$EngDispl^2

set.seed(1)
lm2Fit <- train(FE ~ EngDispl + ED2, 
                data = cars2010,
                method = "lm", 
                trControl = trainControl(method= "cv"))
lm2Fit

## Finally a MARS model (via the earth package)

library(earth)
set.seed(1)
marsFit <- train(FE ~ EngDispl, 
                 data = cars2010,
                 method = "earth",
                 tuneLength = 15,
                 trControl = trainControl(method= "cv"))
marsFit

plot(marsFit)

## Predict the test set data
cars2011$lm1  <- predict(lm1Fit,  cars2011)
cars2011$lm2  <- predict(lm2Fit,  cars2011)
cars2011$mars <- predict(marsFit, cars2011)

## Get test set performance values via caret's postResample function

postResample(pred = cars2011$lm1,  obs = cars2011$FE)
postResample(pred = cars2011$lm2,  obs = cars2011$FE)
postResample(pred = cars2011$mars, obs = cars2011$FE)

### Section 2.2 Themes (No code!) ####
### Section 2.3 Summary (No code!) ####

################################################################################
### Session Information

sessionInfo()

q("no")


