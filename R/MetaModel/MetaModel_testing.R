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
sd <- 2
minLoss <- sd^2
thetas <- lapply(1:M, function(x) {max(rnorm(1,1,0.3),0)})

f <- function(x,theta){
  # x %*% theta
  # x^theta
  # matrix(apply(x^theta,1,sum),ncol=1)
  sin(x*theta)
}
DGP <- function(N, theta, xtype){
  x <- matrix(rnorm(N*K,2,1),ncol=K)
  y <- f(x,theta) + rnorm(nrow(x),0,sd)
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


# i <- 1
# ggplot(data.frame(x=DataValidation[[i]]$x, y=DataValidation[[i]]$y), aes(x=x,y=y))+
#   geom_point(alpha=.1)+
#   geom_point(data=data.frame(x=DataTrain[[i]]$x,y=DataTrain[[i]]$y), aes(x=x,y=y), colour="red")+
#   geom_line(data=data.frame(x=DataValidation[[i]]$x, y=f(DataValidation[[i]]$x, thetas[[i]])))+
#   ggtitle(round(thetas[[i]], 4))

# baseModel --------------------------------------------------------------
# R <- 10
# out <- rep(NA,R)
# for(r in 1:R){
  
inputSize <- K
# layerSizes <- c(1)
layerSizes <- c(64,8,1)
# layerSizes <- c(8,1)
layerDropouts <- c(rep(0.1, length(layerSizes)-1),0)
# layerDropouts <- NULL
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {x}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
baseModel = prepareBaseModel(baseModel,x = train$x)

lr = 0.001
weight_decay = 0
fit <- trainModel(model = baseModel, criterion, train = train[1:2], test = test[1:2], validation = validation[1:2], epochs = 500, minibatch = 100, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr, weight_decay = weight_decay)
# fit <- trainModel(model = baseModel, criterion, train = train[1:2], test = NULL, validation = NULL, epochs = 15, minibatch = 100, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr, weight_decay = weight_decay)
baseModel <- fit$model
baseModelProgress <- fit$progress
y_pred_base <- baseModel(validation$x)
loss_validation_base <- as.array(criterion(y_pred_base,validation$y))
message(round(loss_validation_base,4))
# out[r] <- round(loss_validation_base,4)
# }
# mean(out)

# metaModel ---------------------------------------------------------------
# R <- 10
# out <- rep(NA,R)
# for(r in 1:R){
metaModel <- MetaModel(baseModel, train$xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0)
minibatch <- function() {minibatchSampler(100,train$xtype)}

lr = 0.01
weight_decay = 0
# weight_decay = 1e-5
# weight_decay = 10
fit <- trainModel(model = metaModel, criterion, train, test, validation, epochs = 200, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr , weight_decay = weight_decay)
metaModel <- fit$model
metaModelProgress <- fit$progress
y_pred_meta <- metaModel(validation$x, validation$xtype)
loss_validation_meta <- as.array(criterion(y_pred_meta,validation$y))
message(str_c(round(loss_validation_meta,4), " ", round((loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss),4)))
# out[r] <- round((loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss),4)
# }
# mean(out)

# mesaModels --------------------------------------------------------------
J <- M
losses_validation_mesa <- rep(NA,J)
state_mesa <- rep(NA,J)
for(j in 1:J){
  if(j %% 10 == 0){
    print(str_c("j: ", j, " Time:", Sys.time()))
  }
  mesaModel <- metaModel$MesaModel(metaModel)()
  rows_train <- train$xtype$indices()[2,] == (j-1)
  x_train_subset <- train$x[rows_train,]
  y_train_subset <- train$y[rows_train,]
  rows_test <- test$xtype$indices()[2,] == (j-1)
  x_test_subset <- test$x[rows_test,]
  y_test_subset <- test$y[rows_test,]
  rows_validation <- validation$xtype$indices()[2,] == (j-1)
  x_validation_subset <- validation$x[rows_validation,]
  y_validation_subset <- validation$y[rows_validation,]
  
  fit <- trainModel(model = mesaModel, criterion, list(y_train_subset, x_train_subset), epochs = 50, minibatch = Inf, tempFilePath = NULL, patience = Inf, printEvery = Inf, lr=0.02)
  mesaModel <- fit$model
  
  y_pred_mesa <- mesaModel(x_validation_subset)
  losses_validation_mesa[j] <- as.array(criterion(y_pred_mesa,y_validation_subset))
  state_mesa[j] <- as.array(mesaModel$state_dict()$mesaState)
}
loss_validation_mesa <- mean(losses_validation_mesa)
loss_validation_mesa

# analysis ----------------------------------------------------------------

message(round(loss_validation_base,4))
message(round(loss_validation_meta,4))
message(round(loss_validation_mesa,4))
# (loss_validation_base - minLoss)/minLoss
# (loss_validation_meta - minLoss)/minLoss
# (loss_validation_mesa - minLoss)/minLoss
message(round((loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss),4))
message(round((loss_validation_mesa - loss_validation_base)/(loss_validation_base - minLoss),4))


mesaStatesMeta <- as.vector(as.array(metaModel$state_dict()$mesaLayerWeight))
mesaStatesMesa <- state_mesa 
ggplot(data.table(mesaStatesMeta,mesaStatesMesa,theta = do.call(c,thetas)),aes(x=mesaStatesMeta, y=mesaStatesMesa, colour=theta))+
  geom_point()+
  coord_fixed()+
  geom_abline(intercept = 0, slope = 1) + 
  scale_colour_gradient(low = "red", high = "green")

