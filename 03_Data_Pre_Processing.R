################################################################################
### R code from Applied Predictive Modeling (2013) by Kuhn and Johnson.
### Copyright 2013 Kuhn and Johnson
### Web Page: http://www.appliedpredictivemodeling.com
### Contact: Max Kuhn (mxkuhn@gmail.com)
###
### Chapter 3: Data Pre-Processing
###
### Required packages: AppliedPredictiveModeling, e1071, caret, corrplot
###
### Data used: The (unprocessed) cell segmentation data from the
###            AppliedPredictiveModeling package.
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
### Section 3.1 Case Study: Cell Segmentation in High-Content Screening ####

library(AppliedPredictiveModeling)
data(segmentationOriginal)

library(mice)
md.pattern(segmentationOriginal)

md.pattern(nhanes) # No missing values in this dataset

library(VIM)
aggr(segmentationOriginal)

aggr(nhanes)

help(segmentationOriginal)
head(segmentationOriginal, n = 10L)
names(segmentationOriginal)

## Retain the original training set
segTrain <- subset(segmentationOriginal, Case == "Train")
table(segmentationOriginal$Case)

## Remove the first three columns (identifier columns)
segTrainX <- segTrain[, -(1:3)]
segTrainClass <- segTrain$Class
length(segTrainClass) # 1009

# 移除數個一元、二元或三元的"status"欄位 (Remove some binary or triary columns related to "status")
statusColNum <- grep("Status", names(segTrainX))
statusColNum
# make sure they are unary, binary, or trinary
sapply(segTrainX[, statusColNum], table)
segTrainXNC <- segTrainX[, -statusColNum] # NC: no categorical vars.

## 移除變異為0和近乎為0者 (Removing zero variance varibales or near-zero-variance variables)
library(caret) # for function 'nearZeroVar'
head(nearZeroVar(segTrainX, saveMetrics=TRUE))
# 變異為0的變數 (Identify zero variance variable)
names(segTrainX)[nearZeroVar(segTrainX, saveMetrics=TRUE)$zeroVar]
# remove two zero-variance variables
segTrainXV <- segTrainX[!nearZeroVar(segTrainX, saveMetrics=TRUE)$zeroVar] # XV: without zero variance variables

# 可進⼀步過濾變異近乎為零的變數 (Filtering the near-zero-variance variables)
nearZeroVar(segTrainXV) # 68: "KurtIntenStatusCh1"
names(segTrainXV)[nearZeroVar(segTrainXV)] # 68: "KurtIntenStatusCh1"
table(segTrainXV$KurtIntenStatusCh1) # 966 (0) against 43(1)
segTrainXV <- segTrainXV[-nearZeroVar(segTrainXV)]

################################################################################
### Section 3.2 Data Transformations for Individual Predictors ####

## The column VarIntenCh3 measures the standard deviation of the intensity
## of the pixels in the actin filaments (肌動蛋白絲)

max(segTrainX$VarIntenCh3)/min(segTrainX$VarIntenCh3) # 870.8872 is much greater than 20!

library(e1071) # for skewness()
skewness(segTrainX$VarIntenCh3) # 2.391624

library(caret)

## Use caret's preProcess function to transform for skewness
?preProcess
segPP <- preProcess(segTrainX, method = "BoxCox") # A kind of power transformation
str(segPP)
names(segPP)
segPP$yj
segPP$et
segPP$invHyperbolicSine
segPP$mean
segPP$std
segPP$rotation
segPP$knnSummary
segPP$bagImp
class(segPP)
help(preProcess)

## Apply the transformations
segTrainTrans <- predict(segPP, segTrainX) # predict(model_object模型, new_data數據)

## Results for a single predictor
segPP$bc$VarIntenCh3 # there are 47 "bc"s
### Fig. 3.2 ####
#op <- par(mfrow=c(1,2)) # it's only for pkg {grahics}, not apllied in pkg {lattice}.
histogram(~segTrainX$VarIntenCh3,
          xlab = "Natural Units",
          type = "count")

histogram(~log(segTrainX$VarIntenCh3),
          xlab = "Log Units",
          ylab = " ",
          type = "count")
#par(op)

segPP$bc$PerimCh1

### Fig. 3.3 ####
histogram(~segTrainX$PerimCh1,
          xlab = "Natural Units",
          type = "count")

histogram(~segTrainTrans$PerimCh1,
          xlab = "Transformed Data",
          ylab = " ",
          type = "count")

################################################################################
### Section 3.3 Data Transformations for Multiple Predictors ####

## R's prcomp (PRinciple COMPonent analysis) is used to conduct PCA
pr <- prcomp(~ AvgIntenCh1 + EntropyIntenCh1, 
             data = segTrainTrans, 
             scale. = TRUE) # formula: a formula with no response variable, referring only to numeric variables.

transparentTheme(pchSize = .7, trans = .3) # bookTheme {AppliedPredictiveModeling}: Two lattice themes used throughout the book.

### Fig. 3.4 ####

### Fig. 3.5 ####
# bookTheme(set = TRUE)
xyplot(AvgIntenCh1 ~ EntropyIntenCh1,
       data = segTrainTrans, # 用Box-Cox轉換後的資料集
       groups = segTrain$Class,
       xlab = "Channel 1 Fiber Width",
       ylab = "Intensity Entropy Channel 1",
       auto.key = list(columns = 2),
       #type = c("p", "g"),
       main = "Original Data",
       aspect = 1)

xyplot(PC2 ~ PC1,
       data = as.data.frame(pr$x),
       groups = segTrain$Class,
       xlab = "Principal Component #1",
       ylab = "Principal Component #2",
       main = "Transformed",
       xlim = extendrange(pr$x),
       ylim = extendrange(pr$x),
       type = c("p", "g"),
       aspect = 1)


## Apply PCA to the entire set of predictors.

## There are a few predictors with only a single value, so we remove these first
## (since PCA uses variances, which would be zero)

isZV <- apply(segTrainX, 2, function(x) length(unique(x)) == 1) # indices for zero variance variable
which(isZV)
segTrainX <- segTrainX[, !isZV] # two less variables, same as above
identical(segTrainX, segTrainXV) # TRUE -> FALSE

segPP <- preProcess(segTrainX, c("BoxCox", "center", "scale"))
segTrainTrans <- predict(segPP, segTrainX)

segPCA <- prcomp(segTrainTrans, center = TRUE, scale. = TRUE)

## Plot a scatterplot matrix of the first three components
transparentTheme(pchSize = .8, trans = .3)

### Fig. 3.7 ####
panelRange <- extendrange(segPCA$x[, 1:3])
splom(as.data.frame(segPCA$x[, 1:3]), # Scatter Plot Matrices in {lattice}
      groups = segTrainClass,
      type = c("p", "g"),
      as.table = TRUE,
      auto.key = list(columns = 2),
      prepanel.limits = function(x) panelRange)

## Format the rotation values for plotting
class(segPCA$rotation[, 1:3])
segRot <- as.data.frame(segPCA$rotation[, 1:3])

## Derive the channel variable
vars <- rownames(segPCA$rotation)
channel <- rep(NA, length(vars))
channel[grepl("Ch1$", vars)] <- "Channel 1" # regular expression by $
channel[grepl("Ch2$", vars)] <- "Channel 2"
channel[grepl("Ch3$", vars)] <- "Channel 3"
channel[grepl("Ch4$", vars)] <- "Channel 4"

segRot$Channel <- channel
segRot <- segRot[complete.cases(segRot),] # two less obs.
segRot$Channel <- factor(as.character(segRot$Channel))

## Plot a scatterplot matrix of the first three rotation variables

transparentTheme(pchSize = .8, trans = .7)
panelRange <- extendrange(segRot[, 1:3])
# library(ellipse)
# 
# upperp <- function(...)
#   {
#     args <- list(...)
#     circ1 <- ellipse(diag(rep(1, 2)), t = .1)
#     panel.xyplot(circ1[,1], circ1[,2],
#                  type = "l",
#                  lty = trellis.par.get("reference.line")$lty,
#                  col = trellis.par.get("reference.line")$col,
#                  lwd = trellis.par.get("reference.line")$lwd)
#     circ2 <- ellipse(diag(rep(1, 2)), t = .2)
#     panel.xyplot(circ2[,1], circ2[,2],
#                  type = "l",
#                  lty = trellis.par.get("reference.line")$lty,
#                  col = trellis.par.get("reference.line")$col,
#                  lwd = trellis.par.get("reference.line")$lwd)
#     circ3 <- ellipse(diag(rep(1, 2)), t = .3)
#     panel.xyplot(circ3[,1], circ3[,2],
#                  type = "l",
#                  lty = trellis.par.get("reference.line")$lty,
#                  col = trellis.par.get("reference.line")$col,
#                  lwd = trellis.par.get("reference.line")$lwd)
#     panel.xyplot(args$x, args$y, groups = args$groups, subscripts = args$subscripts)
# } # for the concentric circles !

splom(~segRot[, 1:3],
      groups = segRot$Channel,
      lower.panel = function(...){}, # upper.panel = upperp, # lower.panel left blank !
      prepanel.limits = function(x) panelRange, # same as above
      auto.key = list(columns = 2))

################################################################################
### Section 3.4 Dealing with Missing Values (No code!) ####
### Section 3.5 Removing Variables ####

## To filter on correlations, we first get the correlation matrix for the 
## predictor set

segCorr <- cor(segTrainTrans)

library(corrplot)
corrplot(segCorr, order = "hclust", tl.cex = .35)

## caret's findCorrelation function is used to identify columns to remove.
highCorr <- findCorrelation(segCorr, .75)
length(highCorr) # 43

################################################################################
### Section 3.6 Adding Predictors (No code!) ####
### Section 3.7 Binning Predictors (No code!) ####
### Section 3.8 Computing (Creating Dummy Variables) ####

data(cars)
# load("~/cstsouMac/RandS/Rexamples/AppliedPredictiveModeling/cars.RData")
type <- c("convertible", "coupe", "hatchback", "sedan", "wagon")
cars$Type <- factor(apply(cars[, 14:18], 1, function(x) type[which(x == 1)])) # 804 * 5 -> extract row -> c(1, 0, 0, 0, 0) vs. c("convertible", "coupe", "hatchback", "sedan", "wagon"), so smart!

carSubset <- cars[sample(1:nrow(cars), 20), c(1, 2, 19)]

head(carSubset)
levels(carSubset$Type)

simpleMod <- dummyVars(~ Mileage + Type, # dummyVars {caret}: Create A Full Set of Dummy Variables
                       data = carSubset,
                       ## Remove the variable name from the
                       ## column name
                       levelsOnly = TRUE)
simpleMod
predict(simpleMod, head(carSubset))

withInteraction <- dummyVars(~ Mileage + Type + Mileage:Type,
                             data = carSubset,
                             levelsOnly = TRUE)
withInteraction
predict(withInteraction, head(carSubset))



################################################################################
### Session Information

sessionInfo()

q("no")


