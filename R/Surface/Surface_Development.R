library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls());gc()
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/MetaModel/MetaModel.R")
source("R/Surface/Surface.R")

K <- 1
NTrain <- 100
NTest <- 10
NValidation <- 5000
M <- 100
sd <- 2
minLoss <- sd^2
thetas <- lapply(1:M, function(x) {max(rnorm(1,1,0.2),0)})

f <- function(x,theta){
  # x %*% theta
  # x^theta
  # matrix(apply(x^theta,1,sum),ncol=1)
  sin(x*theta) + x^2
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





# base model ------------------------------------------------------------------------------------------------------------------------------------

inputSize <- K
# layerSizes <- c(1)
layerSizes <- c(64,8,1)
layerSizes <- c(32,8,1)
# layerSizes <- c(8,1)
layerDropouts <- c(rep(0, length(layerSizes)-1),0)
# layerDropouts <- NULL
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_leaky_relu), list(function(x) {x}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
baseModel <- prepareBaseModel(baseModel, train$x)

lr = 0.001
weight_decay = 0
fit <- trainModel(model = baseModel, criterion, train = train[1:2], test = test[1:2], validation = validation[1:2], epochs = 500, minibatch = 100, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr, weight_decay = weight_decay)
baseModel <- fit$model
baseModelProgress <- fit$progress
y_pred_base <- baseModel(validation$x)
loss_validation_base <- as.array(criterion(y_pred_base,validation$y))
message(round(loss_validation_base,4))



metaModel <- MetaModel(baseModel, train$xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0)
minibatch <- function() {minibatchSampler(10,train$xtype)}
lr = 0.001
weight_decay = 0
# weight_decay = 1e-5
# weight_decay = 10
fit <- trainModel(model = metaModel, criterion, train, test, validation, epochs = 200, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr , weight_decay = weight_decay)
metaModel <- fit$model
metaModelProgress <- fit$progress
y_pred_meta <- metaModel(validation$x, validation$xtype)
loss_validation_meta <- as.array(criterion(y_pred_meta,validation$y))
message(str_c(round(loss_validation_meta,4), " ", round((loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss),4)))


# Fourier surface -------------------------------------------------------------------------------------------------------------------------------

# layerSizes <- c(64,8,1)
layerSizes <- c(100,100,1)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x)  nnf_leaky_relu), list(function(x) {x}))
in_Ns <- c(10, 5, 10)
out_Ns <- c(10, 5, 10)
baseModel <- constructFFNSF(inputSize, layerSizes, layerTransforms, in_Ns, out_Ns)
baseModel <- prepareBaseModel(baseModel, train$x)
lr = 0.0005
weight_decay = 0
fit <- trainModel(model = baseModel, criterion, train = train[1:2], test = test[1:2], validation = validation[1:2], epochs = 500, minibatch = 100, tempFilePath = tempFilePath, patience = 10, printEvery = 1, lr = lr, weight_decay = weight_decay)
baseModel <- fit$model
baseModelProgress <- fit$progress
y_pred_base <- baseModel(validation$x)
loss_validation_base <- as.array(criterion(y_pred_base,validation$y))
message(round(loss_validation_base,4))


metaModel <- MetaModel(baseModel, train$xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0)
minibatch <- function() {minibatchSampler(10,train$xtype)}
lr = 0.001
weight_decay = 0
# weight_decay = 1e-5
# weight_decay = 10
fit <- trainModel(model = metaModel, criterion, train, test, validation, epochs = 200, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr , weight_decay = weight_decay)
metaModel <- fit$model
metaModelProgress <- fit$progress
y_pred_meta <- metaModel(validation$x, validation$xtype)
loss_validation_meta <- as.array(criterion(y_pred_meta,validation$y))
message(str_c(round(loss_validation_meta,4), " ", round((loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss),4)))


# Bilinear surface -------------------------------------------------------------------------------------------------------------------------------

# layerSizes <- c(64,8,1)
layerSizes <- c(100,100,1)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x)  nnf_leaky_relu), list(function(x) {x}))
types <- c("DC", rep("CC", length(layerSizes)-2), "CD")
in_Ns <- c(10, 20, 10)
out_Ns <- c(40, 10, 10)
baseModel <- constructFFNSB(inputSize, layerSizes, layerTransforms, in_Ns, out_Ns)
baseModel <- prepareBaseModel(baseModel, train$x)
lr = 0.0005
weight_decay = 0
fit <- trainModel(model = baseModel, criterion, train = train[1:2], test = test[1:2], validation = validation[1:2], epochs = 500, minibatch = 100, tempFilePath = tempFilePath, patience = 10, printEvery = 1, lr = lr, weight_decay = weight_decay)
baseModel <- fit$model
baseModelProgress <- fit$progress
y_pred_base <- baseModel(validation$x)
loss_validation_base <- as.array(criterion(y_pred_base,validation$y))
message(round(loss_validation_base,4))


metaModel <- MetaModel(baseModel, train$xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0)
minibatch <- function() {minibatchSampler(10,train$xtype)}
lr = 0.0001
weight_decay = 0
# weight_decay = 1e-5
# weight_decay = 10
fit <- trainModel(model = metaModel, criterion, train, test, validation, epochs = 200, minibatch = minibatch, tempFilePath = tempFilePath, patience = 10, printEvery = 1, lr = lr , weight_decay = weight_decay)
metaModel <- fit$model
metaModelProgress <- fit$progress
y_pred_meta <- metaModel(validation$x, validation$xtype)
loss_validation_meta <- as.array(criterion(y_pred_meta,validation$y))
message(str_c(round(loss_validation_meta,4), " ", round((loss_validation_meta - loss_validation_base)/(loss_validation_base - minLoss),4)))








# weight <- baseModel$state_dict()$layer_3.weightPar[1,1,,]
# md <- melt(as.data.table(as.array(weight))[,Out := 1:.N], id.vars = "Out")
# md[,In := as.numeric(str_sub(variable,2,-1))]
# ggplot(md,aes(x=In,y=reorder(Out, -Out),fill=value))+
#   geom_tile()+ 
#   geom_text(aes(label = round(value,3)), size=2.5)+
#   scale_fill_gradient2(low = "red",mid = "white",high = "blue") +coord_fixed() 










# constructFFNN = nn_module(
#   initialize = function(inputSize, layerSizes, layerTransforms, layerDropouts = NULL, layerTypes = NULL, in_Ns = NULL, out_Ns = NULL) {
#     self$layerSizes <- layerSizes
#     self$layerTransforms <- layerTransforms
#     self$layerSizesAll <- c(inputSize, layerSizes)
#     self$Dropout <- !is.null(layerDropouts)
#     if(is.null(layerTypes)){
#       layerTypes <- rep("nn_linear", length(layerSizes))
#     }
#     for(i in seq_along(self$layerSizes)){
#       self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
#     }
#     if(self$Dropout){
#       for(i in seq_along(self$layerSizes)){
#         self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
#       }
#     }
#   },
#   forward = function(x) {
#     for(i in seq_along(self$layerSizes)){
#       x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
#       if(self$Dropout){
#         x <- self[[str_c("layerDropout_",i)]](x)
#       }
#     }
#     x
#   },
#   fforward = function(x,state){
#     for(i in seq_along(self$layerSizes)){
#       x <- self$layerTransforms[[i]](nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
#     }
#     x
#   }
# )


# 
# 
# md <- melt(as.data.table(as.array(weight))[,Out := 1:.N], id.vars = "Out")
# md[,In := as.numeric(str_sub(variable,2,-1))]
# ggplot(md,aes(x=In,y=Out,fill=value))+
#   geom_tile()+ 
#   scale_fill_gradient2(low = "red",mid = "white",high = "blue") +coord_fixed()
# 
# 


# nn_surfaceBilinear <- nn_module(
#   initialize = function(in_features, out_features, in_N, out_N, bias = T) {
#     initRange <- 1/sqrt(in_features)
#     self$in_features <- in_features
#     self$out_features <- out_features
#     self$in_N <- min(in_N, in_features)
#     self$out_N <- min(out_N, out_features)
#     self$includeBias <- bias
#     
#     self$weightPar <- nn_parameter(torch_tensor(matrix(runif(self$in_N * self$out_N, -initRange, initRange), ncol=self$in_N, nrow = self$out_N))$unsqueeze(1)$unsqueeze(1))
#     if(self$includeBias){
#       self$biasPar <- nn_parameter(torch_tensor(matrix(runif(self$out_N, -initRange, initRange), ncol=1, nrow = self$out_N))$unsqueeze(1)$unsqueeze(1))
#     }
#   },
#   forward = function(input) {
#     weight <- nnf_interpolate(self$weightPar ,size = c(self$out_features, self$in_features), mode = "bilinear", align_corners = T)[1,1,,]
#     if(self$includeBias){
#       bias <- nnf_interpolate(self$biasPar ,size = c(self$out_features, 1), mode = "bilinear", align_corners = T)[1,1,,1]
#     }else{
#       bias <- NULL
#     }
#     nnf_linear(input, weight = weight, bias = bias)
#   }
# )
# 
# 
# in_features = 10
# out_features =20
# in_N = 2
# out_N = 3
# bias = T
# 
# temp <- nn_surfaceBilinear(in_features, out_features, in_N, out_N, bias = T)
# input <- torch_tensor(matrix(rnorm(in_features * 1),nrow=1))
# temp(input)
# 
# 
# initRange <- 1/sqrt(in_features)
# self <- list()
# 
# self$weightPar <- nn_parameter(torch_tensor(matrix(runif(in_N * out_N, -initRange, initRange), ncol=in_N, nrow = out_N))$unsqueeze(1)$unsqueeze(1))
# self$biasPar <- nn_parameter(torch_tensor(matrix(runif(out_N, -initRange, initRange), ncol=1, nrow = out_N))$unsqueeze(1)$unsqueeze(1))
# 
# 
# weight <- nnf_interpolate(self$weightPar ,size = c(out_features, in_features), mode = "bilinear", align_corners = T)[1,1,,]
# bias <- nnf_interpolate(self$biasPar ,size = c(out_features, 1), mode = "bilinear", align_corners = T)[1,1,,1]
# 
# 
# md <- melt(as.data.table(as.array(weight))[,Out := 1:.N], id.vars = "Out")
# md[,In := as.numeric(str_sub(variable,2,-1))]
# ggplot(md,aes(x=In,y=reorder(Out, -Out),fill=value))+
#   geom_tile()+ 
#   geom_text(aes(label = round(value,3)), size=2.5)+
#   scale_fill_gradient2(low = "red",mid = "white",high = "blue") +coord_fixed() 
# 
# 
# 
# 
# 
# 
# input <- torch_rand(1, 1, 10, 2)
# iinput <- nnf_interpolate(input ,size = c(20,20), mode = "bilinear", align_corners = T)
# as.array(input[1,1,,])
# weight <- as.array(iinput[1,1,,])
# 
# md <- melt(as.data.table(as.array(weight))[,Out := 1:.N], id.vars = "Out")
# md[,In := as.numeric(str_sub(variable,2,-1))]
# ggplot(md,aes(x=In,y=reorder(Out, -Out),fill=value))+
#   geom_tile()+ 
#   geom_text(aes(label = round(value,3)), size=2.5)+
#   scale_fill_gradient2(low = "red",mid = "white",high = "blue") +coord_fixed() 
# 
# 
# 
# 
# 
# 
# 
# 














# fourrier = function(points, theta0, thetac, thetas){
#   M <- thetac$size(1)
#   N <- thetac$size(2)
#   x <- torch_tensor(matrix(seq(0, 2 * pi, length.out = points + 1)[1:points],ncol = points, nrow = M, byrow = T), dtype = torch_float())
#   out <- theta0$unsqueeze(2) + Reduce("+",lapply(1:N, function(n) thetac[,n]$unsqueeze(2) * torch_cos(x*n) + thetas[,n]$unsqueeze(2) * torch_sin(x*n)))
#   return(out)
# }
# 
# nn_surface <- nn_module(
#   initialize = function(in_features, out_features, in_N, out_N, type, bias = T) {
#     initRange <- .01
#     self$in_features <- in_features
#     self$out_features <- out_features
#     self$in_N <- in_N
#     self$out_N <- out_N
#     self$type <- type
#     self$includeBias <- bias
#     switch (type,
#             "CD" = {
#               self$theta0 <- nn_parameter(torch_tensor(rep(0, out_features)))
#               self$thetac <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
#               self$thetas <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
#               if(self$includeBias == T){
#                 self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
#               }
#             },
#             "CC" = {
#               if(self$includeBias == T){
#                 M <- (1 + self$in_N * 2) + 1
#               }else{
#                 M <- (1 + self$in_N * 2)
#               }
#               self$theta0 <- nn_parameter(torch_tensor(rep(0, M)))
#               self$thetac <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
#               self$thetas <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
#             },
#             "DC" = {
#               if(self$includeBias == T){
#                 M <- self$in_features + 1
#               }else{
#                 M <- self$in_features
#               }
#               self$theta0 <- nn_parameter(torch_tensor(rep(0, M)))
#               self$thetac <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
#               self$thetas <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
#             }
#     )
#   },
#   forward = function(input) {
#     switch (self$type,
#             "CD" = {
#               weight <- fourrier(self$in_features, self$theta0, self$thetac, self$thetas)
#               bias <- self$bias
#             },
#             "CC" = {
#               temp <- fourrier(self$out_features, self$theta0, self$thetac, self$thetas)$transpose(1,2)
#               theta0 <- temp[,1]
#               thetac <- temp[,(1 + 1):(1 + self$in_N)]
#               thetas <- temp[,(1 + self$in_N + 1):(1 + self$in_N + self$in_N)]
#               weight <- fourrier(self$in_features, theta0, thetac, thetas)
#               if(self$includeBias == T){
#                 bias <- temp[,(1 + 2 * self$in_N + 1)]
#               }else{
#                 bias <- NULL
#               }
#             },
#             "DC" = {
#               temp <- fourrier(self$out_features, self$theta0, self$thetac, self$thetas)$transpose(1,2)
#               weight <- temp[,1:self$in_features]
#               if(self$includeBias == T){
#                 bias <- temp[,(self$in_features + 1)]
#               }else{
#                 bias <- NULL
#               }
#             }
#     )
#     nnf_linear(input, weight = weight, bias = self$bias)
#   }
# )





















N <- 10
points <- 20
y <- rnorm(5,0,1)
yposition <- seq(0,1,length.out = length(y))
x <- seq(0,1,length.out = points)


nnf_interpolate(torch_tensor(y),size = )

y <- matrix(rnorm(10),5)
input <- torch_tensor(y)
input <- input$unsqueeze(3)
yint <- nnf_interpolate(input,size = c(1,10,1))
as.array(yint)



















x <- train$x
junk <- nn_surface(1,64,2,2,"DC")
junk(x)








nn_linearAlt <- nn_module(
  initialize = function(in_features, out_features, bias = T) {
    initRange <- 1/sqrt(in_features)
    self$weight = nn_parameter(torch_tensor(matrix(runif(in_features * out_features, -initRange, initRange), nrow = out_features, ncol = in_features)))
    if(bias){
      self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
    }else{
      self$bias = NULL
    }
  },
  forward = function(input) {
    nnf_linear(input, weight = self$weight, bias = self$bias)
  }
)


N <- 10
theta0=0
thetac=runif(N,-1,1)
thetas=runif(N,-1,1)

x <- seq(0,2*pi,0.01)
y <- theta0 + Reduce("+",lapply(1:N, function(n) thetac[n] * sin(x*n) + thetas[n] * cos(x*n) ))
plot(x,y,type = "l")



N <- 5
M <- 2
theta0=torch_tensor(rep(0,M))
thetac=torch_tensor(matrix(runif(N * M,-.1,.1),ncol = N))
thetas=torch_tensor(matrix(runif(N * M,-.1,.1),ncol = N))
points <- 10

fourrier = function(points, theta0, thetac, thetas){
  M <- thetac$size(1)
  N <- thetac$size(2)
  x <- torch_tensor(matrix(seq(0, 2 * pi, length.out = points + 1)[1:points],ncol = points, nrow = M, byrow = T), dtype = torch_float())
  out <- theta0$unsqueeze(2) + Reduce("+",lapply(1:N, function(n) thetac[,n]$unsqueeze(2) * torch_cos(x*n) + thetas[,n]$unsqueeze(2) * torch_sin(x*n)))
  return(out)
}

points <- 100
y <- fourrier(points,theta0,thetac,thetas)
plot(as.array(y[1,]))
plot(as.array(y[2,]))




in_features=1
# out_features=3
out_features=50
in_N = 2
out_N = 2
bias = T


type = "DC"
self <- list()
nn_surface <- nn_module(
  initialize = function(in_features, out_features, in_N, out_N, type, bias = T) {
    initRange <- .1
    self$in_features <- in_features
    self$out_features <- out_features
    self$in_N <- in_N
    self$out_N <- out_N
    self$type <- type
    self$includeBias <- bias
    switch (type,
            "CD" = {
              self$theta0 <- nn_parameter(torch_tensor(rep(0, out_features)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
              if(self$includeBias == T){
                self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
              }
            },
            "CC" = {
              if(self$includeBias == T){
                M <- (1 + self$in_N * 2) + 1
              }else{
                M <- (1 + self$in_N * 2)
              }
              self$theta0 <- nn_parameter(torch_tensor(rep(0, M)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
            },
            "DC" = {
              if(self$includeBias == T){
                M <- self$in_features + 1
              }else{
                M <- self$in_features
              }
              self$theta0 <- nn_parameter(torch_tensor(rep(0, M)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(M * in_N, -initRange, initRange), ncol = in_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(M * in_N, -initRange, initRange), ncol = in_N)))
            }
    )
  },
  forward = function(input) {
    switch (self$type,
            "CD" = {
              weight <- fourrier(self$in_features, self$theta0, self$thetac, self$thetas)
              bias <- self$bias
            },
            "CC" = {
              temp <- fourrier(self$out_features, self$theta0, self$thetac, self$thetas)$transpose(1,2)
              theta0 <- temp[,1]
              thetac <- temp[,(1 + 1):(1 + self$in_N)]
              thetas <- temp[,(1 + self$in_N + 1):(1 + self$in_N + self$in_N)]
              weight <- fourrier(self$in_features, theta0, thetac, thetas)
              if(self$includeBias == T){
                bias <- temp[,(1 + 2 * self$in_N + 1)]
              }else{
                bias <- NULL
              }
            },
            "DC" = {
              temp <- fourrier(self$out_features, self$theta0, self$thetac, self$thetas)$transpose(1,2)
              weight <- temp[,1:in_features]
              if(self$includeBias == T){
                bias <- temp[,(in_features + 1)]
              }else{
                bias <- NULL
              }
            }
    )
    nnf_linear(input, weight = weight, bias = self$bias)
  }
)


in_features = 50
# out_features = 5
out_features = 40
in_N = 3
out_N = 2
type = "DC"
bias = T
k <- 2
x <- torch_tensor(matrix(rnorm(in_features * k),nrow=k))
surface <- nn_surface(in_features, out_features, in_N, out_N, type = type, bias = bias)
surface(x)
surface$state_dict()























N <- 2
theta0=torch_tensor(0)
thetac=torch_tensor(runif(N,-1,1))
thetas=torch_tensor(runif(N,-1,1))

fourrier = function(points, theta0, thetac, thetas){
  x <- torch_tensor(seq(0, 2 * pi, length.out = points + 1)[1:points], dtype = torch_float())
  out <- theta0 + Reduce("+",lapply(1:thetac$size(), function(n) thetac[n] * torch_cos(x*n) + thetas[n] * torch_sin(x*n) ))
}

points <- 2
y <- fourrier(points,theta0,thetac,thetas)







y[1]
plot(as.array(y))



torch_cos(theta0 * n)

x <- seq(0,2*pi,0.01)
y <- theta0 + Reduce("+",lapply(1:N, function(n) thetac[n] * torch_cos(x*n) + thetas[n] * torch_sin(x*n) ))
plot(x,y,type = "l")






in_features =1
out_features = 3
bias=T


input <- torch_tensor(matrix(rnorm(10),10,1))
layer <- nn_linear(in_features,out_features)
layer(input)
nnf_linear(input, weight = layer$weight, bias = layer$bias)



self <- list()
initRange <- 1/sqrt(in_features)
self$weight = nn_parameter(torch_tensor(matrix(runif(in_features * out_features, -initRange, initRange), nrow = out_features, ncol = in_features)))
if(bias){
  self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
}else{
  self$bias = NULL
}
nnf_linear(input, weight = self$weight, bias = self$bias)















in_features = 3
out_features = 2
bias = TRUE
junk <- nn_linear(in_features = in_features, out_features = out_features, bias = T)
junk$weight
junk$bias
self <- list()

nn_linearAlt <- nn_module(
  initialize = function(in_features, out_features, bias = T) {
    initRange <- 1/sqrt(in_features)
    self$weight = nn_parameter(torch_tensor(matrix(runif(in_features * out_features, -initRange, initRange), nrow = out_features, ncol = in_features)))
    if(bias){
      self$bias = nn_parameter(torch_tensor(matrix(runif(out_features, -initRange, initRange), nrow = out_features, ncol = 1)))
    }else{
      self$bias = NULL
    }
  },
  forward = function(input) {
    nnf_linear(input, weight = self$weight, bias = bias)
  }
)


junk <- nn_linearAlt(in_features = in_features, out_features = out_features, bias = T)
junk$weight
junk$bias













