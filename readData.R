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

combined.data <- subset(combined.data,Sales != 0)

test.data <- read.csv(file="~/kaggle/rossmann/test.csv")

combined.test <- merge(test.data,store.data,by="Store")

combined.data$Day <- as.numeric(sapply(as.character(combined.data$Date),function(x) substr(x,9,10)))
combined.data$Month <- as.numeric(sapply(as.character(combined.data$Date),function(x) substr(x,6,7)))
combined.data$Year <- as.numeric(sapply(as.character(combined.data$Date),function(x) substr(x,1,4)))

combined.test$Day <- as.numeric(sapply(as.character(combined.test$Date),function(x) substr(x,9,10)))
combined.test$Month <- as.numeric(sapply(as.character(combined.test$Date),function(x) substr(x,6,7)))
combined.test$Year <- as.numeric(sapply(as.character(combined.test$Date),function(x) substr(x,1,4)))

combined.data <- combined.data[,colnames(combined.data)!="Date"]

weights <- 1/combined.data$Sales^2
weights[is.infinite(weights)] <- median(weights[!is.infinite(weights)])

pred.sales <- rep(0,nrow(combined.test))

add.tree <- NULL
set.seed(42)

# seperate structure learning from cross-validation testing
is.struct <- sample(1:nrow(combined.data),size = round(nrow(combined.data)/2))

for (nn in 1:100) {
  cat("Doing iteration:",nn)
  add.tree <- glmTree(Sales~.,input.data = combined.data[,c("Sales",intersect(colnames(combined.data), colnames(combined.test)))],weights=weights,sparse=TRUE,log.base=2,seed=nn,alpha=0,nTrees=15,fitStruct=add.tree,is.struct=is.struct)
  pred.sales <- predict(add.tree,combined.test,s="lambda.min")[,1]

  pred.sales.df <- data.frame(Id=combined.test$Id,Sales=predict(add.tree, combined.test[,intersect(colnames(combined.data), colnames(combined.test))],"lambda.min")[,1])

  write.csv(pred.sales.df,file="~/kaggle/rossmann/20151002.csv",row.names = FALSE)

  gc()
}

