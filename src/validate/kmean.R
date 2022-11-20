# lOADING THE REQUIRED LIBRARIES
library(C50)             # For decision tree induction classification
library(dplyr)           # For data manipulation
library(ggplot2)         # For plotting 
library(plotly)
library(caret)           # for data-preprocessing & for data-partitioning
library(magrittr)
library(fastknn)

library(readr)           # for "read_csv" function
library(stats)           # dependancy fulfilment for dplyr and plotly packages
library(ggthemes)
library(forcats)         # for "fct_reorder" function
library(vcd)             # For association plots
library(mclust)
library(cluster)       # clustering algorithms
library(factoextra)

# Reading the datasets
dataset_sign_mnist_train<-read_csv("../input/sign_mnist_train.csv")
dataset_sign_mnist_test <-read_csv("../input/sign_mnist_test.csv")


# Changing the column label into factors for train and test data
dataset_sign_mnist_train$label<- as.factor(dataset_sign_mnist_train$label)
dataset_sign_mnist_test $label<- as.factor(dataset_sign_mnist_test$label)


# dataset_sign_mnist_train                              # training data
traindata_predicted <- dataset_sign_mnist_train[,-1]    # Excluding column 1
traindata_target    <- dataset_sign_mnist_train[,1]     # Including only column 1


# dataset_sign_mnist_test                               # test data
testdata_predicted <- dataset_sign_mnist_test[,-1]      # Excluding column 1
testdata_target    <- dataset_sign_mnist_test[,1]       # Including only column 1


#function for finding column indexs.
colIndex = function(colName, data){
    return (which(names(data)==colName))
}


# defining getModel() function which takes 2 parameters as inputs Train Data without target values and Target and returns Model as output.
getModel = function(traindata_predicted, traindata_target){
  model <- C5.0(
                  x = traindata_predicted,        # Predictors (only)
                  y = traindata_target,        # Target values
                  trials = 10,                # Boosting helps in generalization
                  # No of boosting steps
                  control = C5.0Control       # parameter control list. May specify separately
                  (
                    noGlobalPruning = FALSE,  # Should global pruning be done? FALSE implies: Yes, do it
                    # FALSE => More generlization
                    CF = 0.15,                # Larger CF (0.75)=>Less tree-pruning.
                    # Smaller values (0.1) =>More tree pruning & more general
                    minCases = 4,             # Min cases per leaf-node.
                    # More cases-> More generalization
                    #sample = 0.80,           # Take 80% sample for training
                    winnow = TRUE,            # TRUE may make it more general, FALSE less general
                    earlyStopping = TRUE      # Should boosting be stopped early?
                  )
                )
 return(model)
}


startTime<-Sys.time()
model <- getModel(traindata_predicted, traindata_target$label)
print("Time taken to create model")
Sys.time()-startTime

#  function getModelAccuracy() for determining accuracy of our Model. This will take Test/Validation data without target, Target of validation/test data and model. 

getModelAccuracy = function(vDataPre,vDataTar, model){
  #  Decision Tree without and Feature Engineering
  #  Make predictions now of type class
  print("-------------------Predicting Output on Validation Data--------------------")
  out <- predict(model, vDataPre , type = "class")
  
  #  Create a dataframe of actual and predicted classes
  #  for quick comparisons
  print("-----------------Creating Actual vs Predicted Matrix---------------------")
  df_comp <- data.frame(predicted = out, actual = vDataTar) 
  
  # Calculate accuracy   
  accuracy <- sum(as.character(df_comp$actual) == as.character(df_comp$predicted)) / nrow(df_comp)
  
  # Ploting accuracy on graph
  tb<-table(pred=df_comp$predicted,actual=df_comp$actual)
  print(assoc(tb, shade = T, labeling = labeling_values))
  return(accuracy)
}
