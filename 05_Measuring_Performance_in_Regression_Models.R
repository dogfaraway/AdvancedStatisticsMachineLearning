################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 5: Measuring Performance in Regression Models
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

### Section 5.1 Quantitative Measures of Performance (No code!) ####
### Section 5.2 The Variance-Bias Trade-off (No code!) ####
### Section 5.3 Computing ####
# Use the 'c' function to combine numbers into a vector
observed <- c(0.22, 0.83, -0.12, 0.89, -0.23, -1.30, -0.15, -1.4, 0.62,  0.99, -0.18, 0.32,  0.34, -0.30,  0.04, -0.87, 0.55, -1.30, -1.15, 0.20)

predicted <- c(0.24, 0.78, -0.66, 0.53, 0.70, -0.75, -0.41, -0.43, 0.49, 0.79, -1.19, 0.06, 0.75, -0.07, 0.43, -0.42, -0.25, -0.64, -1.26, -0.07)

residualValues <- observed - predicted # vectorization 矢/向量化
summary(residualValues)

# Observed values versus predicted values
# It is a good idea to plot the values on a common scale.
axisRange <- extendrange(c(observed, predicted))
plot(observed, predicted, ylim = axisRange, xlim = axisRange)
# Add a 45 degree reference line
abline(0, 1, col = "darkgrey", lty = 2)
# Predicted values versus residuals (RESIDUALS PLOT)
plot(predicted, residualValues, ylab = "residual")
abline(h = 0, col = "darkgrey", lty = 2)

library(caret)
R2(predicted, observed)
RMSE(predicted, observed)

# Simple correlation (Pearson's correlation coefficient)
cor(predicted, observed)

# Rank correlation
cor(predicted, observed, method = "spearman")


