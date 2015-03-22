# Loading packages for parallel computing and setting options
library("cluster")
library("parallel")
library("doSNOW")
coreNumber <- max(detectCores(),1)
cluster <- makeCluster(coreNumber, type = "SOCK",outfile="")
registerDoSNOW(cluster)

# Loading packages for analysis and model building
# library("plyr")
library("dplyr")
library("ggplot2")
library("GGally")
library("caret")

# Loading data into R
print("# Loading Data...")
work.dir <- "C:/Users/Nicholas/Documents/GitHub/Class-8-Course-Project"
setwd(work.dir)
dataFull <- read.csv("./data/pml-training.csv", header = T,
                      stringsAsFactors = T)
dataProb <- read.csv("./data/pml-testing.csv", header = T,
                      stringsAsFactors = T)

# Partitioning full problem data set into training and testing sets (does not
# include the "testing" data set in the project). Train-test split: 7:3
print("# Setting up datasets...")
set.seed(1234)
inTrain <- createDataPartition(y = dataFull$classe, p = 0.7, list = F)
training <- dataFull[inTrain, ]
testing <- dataFull[-inTrain, ]

# Preprocessing training data, removing columns with little or no information,
# as well as corresponding columns that do not contain data in the test data
# set. Subsequently using PCA to collapse variables into 4 vectors (intuitively
# determined due to the number of sensors used in the study).
print("# Preprocessing data...")
trainClean <- select(training, -c(1:7, 12:36, 50:59, 69:83, 87:101,
                                  103:112, 125:139, 141:150))
testClean <- select(testing, -c(1:7, 12:36, 50:59, 69:83, 87:101,
                                 103:112, 125:139, 141:150))
modPC <- preProcess(trainClean[, -length(trainClean)],
                    method = c("pca", "scale", "center"))
trainPC <- predict(modPC, trainClean[, -length(trainClean)])
testPC <- predict(modPC, testClean[, -length(testClean)])
trainPC$classe <- trainClean$classe

# Setting tuning parameters


# Building random forest model
print("# Building RF model...")
library("randomForest")
modfitRF <- randomForest(classe ~ ., data = trainPC)

# # Building Stochastic Gradient Boosting model
# print("# Building GBM model...")
# modfitGBM <- train(classe ~ ., data = trainPC, method = "gbm", verbose = F)
#
# # Building ridge regression model
# print("# Building RR model...")
# modfitRR <- train(classe ~ ., data = trainClean, method = "ridge")

# Building predictions
print("# Building predictions and confusion matrices...")
predRF <- predict(modfitRF, testPC)
# predGBM <- predict(modfitGBM, testing)
# predRR <- predict(modfitRR, testing)

conmatRF <- confusionMatrix(predRF, testing$classe)
# conmatGBM <- confusionMatrix(predGBM, testing$classe)
# conmatRR <- confusionMatrix(predRR, testing$classe)

# # Building ensemble model
# print("# Building ensemble model...")
# trainPredRF <- predict(modfitRF, trainPC)
# trainPredGBM <- predict(modfitGBM, trainPC)
# trainPredRR <- predict(modfitRR, trainClean)
# enDataTrain <- data.frame(rf = trainPredRF, gbm = trainPredGBM,
#                      RR = trainPredRR, classe = trainClean$classe)
# modfitEN <- train(classe ~ ., data = enDataTrain, method = "ridge")
#
# #Building predictions using ensemble model
# print("# Building predictions on ensemble model...")
# enDataTest <- data.frame(rf = predRF, gbm = predGBM, RR = predRR)
# predEN <- predict(modfitEN, enDataTest)
# conmatEN <- confusionMatrix(predEN, testing$classe)

# Preprocessing problem cases
probClean <- select(dataProb, -c(1:7, 12:36, 50:59, 69:83, 87:101,
                                 103:112, 125:139, 141:150, 160))
probPC <- predict(modPC, probClean)

# Predicting outcomes of problem cases
probRF <- predict(modfitRF, probPC)
