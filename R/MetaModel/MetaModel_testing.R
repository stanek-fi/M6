library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls());gc()
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/MetaModel/MetaModel.R")


K <- 1
NTrain <- 100
NTest <- 10
NValidation <- 5000
M <- 100

thetas <- lapply(1:M, function(x) {max(rnorm(1,1,0.03),0)})

f <- function(x,theta){
  # x %*% theta 
  # x^theta
  matrix(apply(x^theta,1,sum),ncol=1)
}
DGP <- function(N, theta, xtype){
  x <- matrix(pmax(rnorm(N*K,2,1),0),ncol=K)
  y <- f(x,theta) + rnorm(nrow(x),0,1)
  xtype <- rep(xtype, nrow(x))
  return(list(
    y = y,
    x = x,
    xtype = xtype
  ))
}
ConvertToTensor <- function(Data){
  y <- torch_tensor(do.call(rbind,lapply(Data, function(d) d$y)))
  x <- torch_tensor(do.call(rbind,lapply(Data, function(d) d$x)))
  xtype_factor <- as.factor(do.call(c,lapply(Data, function(d) d$xtype)))
  i <- torch_tensor(t(cbind(seq_along(xtype_factor),as.integer(xtype_factor))),dtype=torch_int64())
  v <- torch_tensor(rep(1,length(xtype_factor)))
  xtype <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor),length(levels(xtype_factor))))$coalesce()
  return(list(
    y = y,
    x = x,
    xtype = xtype
  ))
}

DataTrain <- lapply(seq_along(thetas), function(i) DGP(NTrain, thetas[[i]], i))
train <- ConvertToTensor(DataTrain)
DataTest <- lapply(seq_along(thetas), function(i) DGP(NTest, thetas[[i]], i))
test <- ConvertToTensor(DataTest)
DataValidation <- lapply(seq_along(thetas), function(i) DGP(NValidation, thetas[[i]], i))
validation <- ConvertToTensor(DataValidation)
criterion = function(y_pred,y) {mean((y_pred-y)^2)}


# baseModel --------------------------------------------------------------

inputSize <- K
# layerSizes <- c(1)
layerSizes <- c(32,8,1)
# layerDropouts <- c(rep(0.0, length(layerSizes)-1),0)
layerDropouts <- NULL
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {x}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
baseModel = prepareBaseModel(baseModel,x = train$x)

fit <- trainModel(model = baseModel, train[1:2], test[1:2], criterion, epochs = 500, minibatch = 100, tempFilePath = tempFilePath, patience = 5, printEvery = 10, lr=0.001)
baseModel <- fit$model
baseModelProgress <- fit$progress
y_pred_base <- baseModel(validation$x)
loss_validation_base <- as.array(criterion(y_pred_base,validation$y))
loss_validation_base


# metaModel ---------------------------------------------------------------

metaModel <- MetaModel(baseModel, train$xtype, mesaParameterSize = 1)
minibatch <- function() {minibatchSampler(10,train$xtype)}

fit <- trainModel(model = metaModel, train, test, criterion, epochs = 50, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr=0.001)
metaModel <- fit$model
metaModelProgress <- fit$progress
y_pred_meta <- metaModel(validation$x, validation$xtype)
loss_validation_meta <- as.array(criterion(y_pred_meta,validation$y))
loss_validation_meta


# analysis ----------------------------------------------------------------
minLoss <- 1
loss_validation_base
loss_validation_meta
(loss_validation_base - minLoss)/minLoss
(loss_validation_meta - minLoss)/minLoss
(loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss)