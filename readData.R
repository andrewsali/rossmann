rm(list=ls())
set.seed(1)
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

pred.sales <- rep(0,nrow(combined.test))
for (nn in 1:1) {
  cat("Doing iteration:",nn)
  test <- glmTree(Sales~.,input.data = combined.data[,c("Sales",intersect(colnames(combined.data), colnames(combined.test)))],weights=weights,sparse=TRUE,log.base=1.5,seed=nn)
  pred.sales <- pred.sales + predict(test,combined.test,s="lambda.min")[,1]
}
pred.sales.df <- data.frame(Id=combined.test$Id,Sales=pred.sales/nn)

write.csv(pred.sales.df,file="~/kaggle/rossmann/20151002.csv",row.names = FALSE)
