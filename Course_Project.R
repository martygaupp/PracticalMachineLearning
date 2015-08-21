# Analysis Assignment

# load the appropriate packages

library(caret)
library(randomForest)
library(gbm)

# to get help on caret go to:
# http://topepo.github.io/caret/index.html

# load the data

FitData <- read.csv("pml-training.csv")
str(FitData)

# classe variable values:
#   A = exactly according to the specification
#   B = throwing the elbows to the front
#   C = lifting the dumbbell only halfway
#   D = lowering the dumbbell only halfway
#   E = throwing the hips to the front

# load just the data with columns
#  I did my adjusting in Excel and then saved this as a .csv file

JustDataCols <- read.csv("JustData.csv")

# partition the data

inTrain <- createDataPartition(y = JustDataCols$classe, p = 0.1, list = FALSE)
training <- JustDataCols[inTrain,]
testing <- JustDataCols[-inTrain,]

# random forest method from caret (rf)

fitControl <- trainControl(method = "cv",
                           number = 3)

ptm <- proc.time()
RFFit <- train(training$classe ~ .,
               data = training,
               method = "rf",
               trControl = fitControl,
               verbose = FALSE)
elapsed <- proc.time() - ptm
RFTime <- round(elapsed[3], 2)
RFPred <- predict(RFFit, testing)
RFAcc <- 100*round(confusionMatrix(testing$classe, RFPred)$overall['Accuracy'], 4)

# boosted tree model from caret (gbm)

ptm <- proc.time()
GBMFit <- train(training$classe ~ .,
                data = training,
                method = "gbm",
                verbose = FALSE)
elapsed <- proc.time() - ptm
GBMTime <- round(elapsed[3], 2)
GBMPred <- predict(GBMFit, testing)
GBMAcc <- 100*round(confusionMatrix(testing$classe, GBMPred)$overall['Accuracy'], 4)

# random forest directly from randomforest package

ptm <- proc.time()
RandForFit <- randomForest(training$classe ~ ., data = training)
elapsed <- proc.time() - ptm
RandForTime <- round(elapsed[3], 2)
RandForPred <- predict(RandForFit, testing)
RandForAcc <- 100*round(confusionMatrix(testing$classe, RandForPred)$overall['Accuracy'], 4)

# Make predictions on the Test Set

TestSetData <- read.csv("TestSet.csv")
TestSetPred <- predict(RFFit, TestSetData)
TestSetPred

# Function for writing the files

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(TestSetPred)

