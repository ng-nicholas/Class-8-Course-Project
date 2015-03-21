# Loading packages for parallel computing and setting options
library("cluster")
library("parallel")
library("doSNOW")
coreNumber <- max(detectCores(),1)
cluster <- makeCluster(coreNumber, type = "SOCK",outfile="")
registerDoSNOW(cluster)

# Loading packages for analysis and model building
library("dplyr")
library("ggplot2")
library("GGally")
library("caret")

# Loading data into R
work.dir <- "C:/Users/Nicholas/Documents/GitHub/Class-8-Course-Project"
setwd(work.dir)
data.full <- read.csv("./data/pml-training.csv", header = T,
                      stringsAsFactors = F)
data.test <- read.csv("./data/pml-testing.csv", header = T,
                      stringsAsFactors = F)

# Partitioning full problem data set into training and testing sets. (does not
# include the "testing" data set in the project)
set.seed(1234)
inTrain <- createDataPartition(y = data.full$classe, p = 0.7, list = F)
training <- data.full[inTrain, ]
testing <- data.full[-inTrain, ]

# Preprocessing training data, removing columns with little or no information,
# as well as corresponding columns that do not contain data in the test data
# set. Subsequently using PCA to collapse variables into 5 vectors (intuitively
# determined due to the number of sensors used in the study).
training.cleaned <- select(training, -c(6, 12:36, 50:59, 69:83, 87:101, 103:112,
                                        125:139, 141:150))
preProc <- preProcess(training[,-c(1:2, 59)],
                      method = c("pca", "scale", "center"),
                      pcaComp = 5)
trainPC <- predict(preProc, training[,-c(1:2, 59)])

modelFit <- train(training$type ~ .,method="glm",data=trainPC)
