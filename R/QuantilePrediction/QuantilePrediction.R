library(data.table)
library(stringr)
library(torch)
library(ggplot2)
rm(list=ls())
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")

Stocks <- readRDS(file.path("Data","StocksM6.RDS"))

TimeEnd <- as.Date("2023-01-08")
TimeStart <- TimeEnd - (7*4) * 1000
TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
TimeBreaks <- TimeBreaks[TimeBreaks>as.Date("1969-12-01")]
TimeBreaksNames <- str_c(TimeBreaks[-length(TimeBreaks)]+1, " : " , TimeBreaks[-1])
  
s <- 1
Stocks <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
  Stock <- Stocks[[s]]
  Ticker <- names(Stocks)[s]
  colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")             
  Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
  Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
  Stock[,Ticker := Ticker]
  Stock
}))

featureList <- list(
  function(SD) {Return(SD)}, #first is just return for y generation
  function(SD) {LagVolatility(SD, lags = 1:5)},
  function(SD) {LagReturn(SD, lags = 1:5)}
)
StocksAggr <- Stocks[,computeFeatures(.SD,featureList),.(Ticker)]
featureNames <- names(StocksAggr)[-(1:3)]
StocksAggr[, ReturnQuintile := computeQuintile(Return), Interval]

StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr[, IntervalStart := as.Date(str_sub(Interval,1,10))]
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]
StocksAggr <- StocksAggr[Ticker %in% unique(StocksAggr$Ticker)[1:100]]

y <- StocksAggr[,ReturnQuintile]
y = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())

x <- StocksAggr[,.SD,.SDcols = featureNames]
x = torch_tensor(as.matrix(x), dtype = torch_float())

xtype_factor <- as.factor(StocksAggr$Ticker)
i <- torch_tensor(t(cbind(seq_along(xtype_factor),as.integer(xtype_factor))),dtype=torch_int64())
v <- torch_tensor(rep(1,length(xtype_factor)))
xtype <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor),length(levels(xtype_factor))))$coalesce()


trainSplit <- 0.7
trainN <- round(nrow(x)*trainSplit)
trainRows <- 1:trainN
testRows <- (trainN+1):nrow(x)

y_train <- y[trainRows,]
x_train <- x[trainRows,]
xtype_train <- subsetSparseTensor(xtype, rows = trainRows)
y_test <- y[testRows,]
x_test <- x[testRows,]
xtype_test <- subsetSparseTensor(xtype, rows = testRows)

inputSize <- length(featureNames)
layerSizes <- c(8, 5)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {nnf_softmax(x,2)}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms)
baseModel = prepareBaseModel(baseModel,x = x_train)
criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}

optimizer = optim_adam(baseModel$parameters, lr = 0.01)
epochs = 100
progress <- data.table(
  epoch = seq_len(epochs),
  loss_train = as.numeric(rep(NA, epochs)),
  loss_test = as.numeric(rep(NA, epochs))
)
start <- Sys.time()
for(i in 1:epochs){
  optimizer$zero_grad()
  y_pred = baseModel(x_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  progress[i, loss_train := loss$item()]
  progress[i, loss_test := as.array(criterion(baseModel(x_test), y_test))]
  if(i %% 10 == 0){
    print(str_c("Epoch:", i," loss_train: ", round(progress[i,loss_train],3)," loss_test:", round(progress[i,loss_test],3), " Time:", Sys.time()))
  }
}
Sys.time() - start 
ggplot(melt(progress,id.vars = "epoch"),aes(x=epoch,y=value,colour=variable))+
  geom_line()




metaModelInicialized <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1)
metaModel <- metaModelInicialized

minibatchSampler = function(batchSize, xtype_train){
  rows <- as.array(xtype_train$indices()[1,]) + 1
  columns <- as.array(xtype_train$indices()[2,]) + 1
  uniqueColumns <- unique(columns)
  bs <- sample(seq_along(uniqueColumns),replace = F)
  bs <- split(bs, ceiling(seq_along(bs)/batchSize))
  bs <- lapply(bs, function(x) uniqueColumns[x])
  bs <- lapply(bs, function(x) which(columns %in% x))
  bs
}
patience <- 5


optimizer = optim_adam(metaModel$parameters, lr = 0.01)
epochs = 20
progress <- data.table(
  epoch = seq_len(epochs),
  loss_train = as.numeric(rep(NA, epochs)),
  loss_test = as.numeric(rep(NA, epochs))
)
modelMemory <- vector(mode = "list", length = patience)

start <- Sys.time()
for(i in 1:epochs){
  minibatches <- minibatchSampler(Inf,xtype_train)
  for(ir in seq_along(minibatches)){
    minibatch <- minibatches[[ir]]
    x_train_mb <- x_train[minibatch,]
    y_train_mb <- y_train[minibatch,]
    xtype_train_mb <- subsetSparseTensor(xtype_train, rows = minibatch)
    
    optimizer$zero_grad()
    y_pred_mb = metaModel(x_train_mb, xtype_train_mb)
    loss = criterion(y_pred_mb, y_train_mb)
    loss$backward()
    optimizer$step()
  }

  loss_train_e <- as.array(criterion(metaModel(x_train,xtype_train), y_train))
  loss_test_e <- as.array(criterion(metaModel(x_test,xtype_test), y_test))
  progress[i, loss_train := loss_train_e]
  progress[i, loss_test := loss_test_e]
  if(i %% 10 == 0){
    print(str_c("Epoch:", i," loss_train: ", round(progress[i,loss_train],3)," loss_test:", round(progress[i,loss_test],3), " Time:", Sys.time()))
  }
}
Sys.time() - start 
ggplot(melt(progress,id.vars = "epoch"),aes(x=epoch,y=value,colour=variable))+
  geom_line()







# mesaModel <- metaModel$MesaModel(metaModel)()
# optimizer = optim_adam(mesaModel$parameters, lr = 0.01)
# epochs = 300
# 
# j <- 8
# rowsubset <- xtype_train$indices()[2,] == (j-1)
# x_train_subset <- x_train[rowsubset,]
# y_train_subset <- y_train[rowsubset]
# for(i in 1:epochs){
#   
#   optimizer$zero_grad()
#   y_pred = mesaModel(x_train_subset)
#   loss = criterion(y_pred, y_train_subset)
#   loss$backward()
#   optimizer$step()
#   
#   # Check Training
#   if(i %% 10 == 0){
#     cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
#   }
# }
# t(as.array(metaModel$state_dict()$mesaLayerWeight))[max(1,(j-1)):(j+1)]
# mesaModel$state_dict()
# 







