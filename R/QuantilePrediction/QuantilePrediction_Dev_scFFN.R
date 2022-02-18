library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls());gc()
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")







# self <- list()
# constructSCNN2 = nn_module(
#   initialize = function(inputSize, layerSizes, layerTransforms, scfun, scsize, sclags, layerDropouts = NULL) {
#     self$layerSizes <- layerSizes
#     self$layerTransforms <- layerTransforms
#     self$layerSizesAll <- c(inputSize, layerSizes)
#     self$Dropout <- !is.null(layerDropouts)
#     self$scfun <- scfun
#     self$scsize <- scsize
#     self$sclags <- sclags
#     self$sclayer <- nn_linear(self$scsize * self$sclags, self$layerSizesAll[length(self$layerSizesAll)])
#     for(i in seq_along(self$layerSizes)){
#       self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
#     }
#     if(self$Dropout){
#       for(i in seq_along(self$layerSizes)){
#         self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
#       }
#     }
#   },
#   forward = function(x,xtype,y) {
#     for(i in seq_along(self$layerSizes)){
#       xout <- self[[str_c("layer_",i)]](x)
#       x <- self$layerTransforms[[i]](xout)
#       if(self$Dropout){
#         x <- self[[str_c("layerDropout_",i)]](x)
#       }
#     }
#     ypred <- x
#     sc <- self$scfun(ypred,y)
#     
#     # sclagged <- torch_cat(lapply(1:self$sclags, function(lg) {
#     #   torch_cat(list(torch_ones(lg,self$scsize) * sc$mean(),sc[1:(sc$size(1)-lg),]), 1)
#     # }), 2)
#     
#     sclagged <- torch_zeros(nrow(x), self$scsize * self$sclags)
#     columns <- xtype$indices()[2,]
#     uniqueColumns <- unique(as.array(columns))
#     for(i in uniqueColumns){
#       indices <- columns == i
#       scsub <- sc[indices,]
#       scsublagged <- torch_cat(lapply(1:self$sclags, function(lg) {
#         torch_cat(list(torch_zeros(lg,self$scsize), sc[1:(scsub$size(1)-lg),]), 1)
#       }), 2)
#       sclagged[indices] <- scsublagged
#     }
#     
#     correction <- self$sclayer(sclagged) * 1e-3
#     ypredsc <- self$layerTransforms[[length(self$layerSizes)]](xout + correction)
#     ypredsc
#   }
# )


# ii <- sample(1:5,1)
# c(rep(0,5-ii),rep(1,ii))



lags <- 3
n_train <- 10000
n_test <- 10000
n <- n_train + n_test
y <- torch_tensor(t(sapply(1:n, function(i) ifelse(1:5>=sample(1:5,1),1,0) )))
# x_train <- torch_tensor(matrix(rnorm(k*n),ncol=k))
x <- torch_cat(lapply(1:lags, function(lg) torch_cat(list(y[1:lg,],y[1:(n-lg),]))),2)

R <- 1
y_train <- torch_cat(lapply(1:R, function(r) y[1:n_train,]),1)
x_train <- torch_cat(lapply(1:R, function(r) x[1:n_train,]),1)
y_test <- torch_cat(lapply(1:R, function(r) y[(n_train +1):n,]),1)
x_test <- torch_cat(lapply(1:R, function(r) x[(n_train +1):n,]),1)


# n <- 1000
# # y_test <- torch_tensor(t(sapply(1:n, function(i) sample(c(1,rep(0,4))))))
# y_test <- torch_tensor(t(sapply(1:n, function(i) ifelse(1:5>=sample(1:5,1),1,0) )))
# # x_test <- torch_tensor(matrix(rnorm(k*n),ncol=k))
# x_test <- torch_cat(list(y_test[1:1,],y_test[1:(n-1),]))

inputSize <- 5 * lags
layerSizes <- c(16, 5)
layerDropouts <- c(rep(0.0, length(layerSizes)-1),0)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_leaky_relu), list(function(x) {nnf_softmax(x,2)}))
scfun <- function(ypred,y) {
  torch_cat(list(
    y
    # ComputeRPSTensorVector(ypred,y)$unsqueeze(2),
    # torch_matmul(ypred, torch_tensor(matrix(-2:2), dtype = torch_float())) - torch_matmul(y, torch_tensor(matrix(-2:2), dtype = torch_float())),
    # torch_matmul(ypred, torch_tensor(abs(matrix(-2:2)), dtype = torch_float())) - torch_matmul(y, torch_tensor(abs(matrix(-2:2)), dtype = torch_float())),
    # ypred - y
    # abs(ypred - y)
  ), 2)
}
sclags <- 1
# scsize <- 13
scsize <- 5 
model <- constructSCNN(inputSize, layerSizes, layerTransforms, scfun = scfun, scsize = scsize, sclags = sclags, layerDropouts = layerDropouts)

# ypred <- model(x_train,y_train)
# ypred
criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}
lr <- 0.001
isSparse <- c(F,F,F)
fit <- trainModel(model = model, criterion, train = list(y_train, x_train, y_train), test = list(y_test, x_test, y_test), validation = NULL, epochs = 2000, minibatch = 100, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr=lr, isSparse = isSparse)
model <- fit$model




y_pred <- model(x_train,y_train)
criterion(y_pred,y_train)

y_pred <- torch_tensor(matrix(0.2,x_train$size(1), 5))
criterion(y_pred,y_train)






self <- list()
constructSCNN <- nn_module(
  initialize = function(inputSize, scfun, scsize, sclags) {
    self$l1 <- nn_linear(inputSize, 3)
    self$l2 <- nn_linear(3, 5)
    self$scfun <- scfun
    self$sclayer <- nn_linear(scsize * sclags, 5)
  },
  forward = function(x,y) {
    n <- nrow(x)
    x <- self$l1(x)
    x <- self$l2(x)
    ypred <- nnf_softmax(x,2)
    
    sc <- self$scfun(ypred,y)
    scsize <- sc$size(2)
    sclagged <- torch_cat(lapply(1:sclags, function(lg) {
      torch_cat(list(torch_ones(lg,scsize) * sc$mean(),sc[1:(n-lg),]), 1)
    }), 2)
    correction <- self$sclayer(sclagged)
    
    nnf_softmax(x + correction, 2)
  }
)



k <- 2
n <- 10
x <- torch_tensor(matrix(rnorm(k*n),ncol=k))
y <- torch_tensor(t(sapply(1:n, function(i) sample(c(1,rep(0,4))))))

inputSize <- k
scfun <- function(ypred,y) {
  torch_cat(list(
    ComputeRPSTensorVector(ypred,y)$unsqueeze(2),
    torch_matmul(ypred, torch_tensor(matrix(-2:2), dtype = torch_float())) - torch_matmul(y, torch_tensor(matrix(-2:2), dtype = torch_float())),
    torch_matmul(ypred, torch_tensor(abs(matrix(-2:2)), dtype = torch_float())) - torch_matmul(y, torch_tensor(abs(matrix(-2:2)), dtype = torch_float())),
    ypred - y,
    abs(ypred - y)
  ), 2)
}
sclags <- 2
scsize <- 13
model <- constructSCNN(k, scfun, scsize, sclags)

ypred <- model(x,y)

ComputeRPSTensor(ypred,y)
ComputeRPSTensorVector(ypred,y)










