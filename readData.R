rm(list=ls())
library(MatrixModels)
library(rpart)
library(glmnet)
library(treeClust)

source("~/kaggle/glmTree/R/bagged.trees.R")
train.data <- read.csv(file="~/kaggle/rossmann/train.csv")
store.data <- read.csv(file="~/kaggle/rossmann/store.csv")

combined.data <- merge(train.data,store.data,by="Store")

sombined.data <- subset(combined.data,Sales != 0)

test.data <- read.csv(file="~/kaggle/rossmann/test.csv")

combined.test <- merge(test.data,store.data,by="Store")

combined.data$Month <- as.numeric(sapply(as.character(combined.data$Date),function(x) substr(x,6,7)))
combined.data$Year <- as.numeric(sapply(as.character(combined.data$Date),function(x) substr(x,1,4)))

combined.test$Month <- as.numeric(sapply(as.character(combined.test$Date),function(x) substr(x,6,7)))
combined.test$Year <- as.numeric(sapply(as.character(combined.test$Date),function(x) substr(x,1,4)))

combined.data <- combined.data[,colnames(combined.data)!="Date"]

weights <- 1/combined.data$Sales^2
weights[is.infinite(weights)] <- median(weights[!is.infinite(weights)])

test <- glmTree(Sales~.,input.data = combined.data[,c("Sales",intersect(colnames(combined.data), colnames(combined.test)))],weights=weights,sparse=TRUE,nTrees=20)

pred.sales <- data.frame(Id=combined.test$Id,Sales=predict(test,combined.test)[,1])

write.csv(pred.sales,file="~/kaggle/rossmann/20151002.csv",row.names = FALSE)
