library(data.table)
library(stringr)
library(torch)
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
StocksAggr <- StocksAggr[Ticker %in% unique(StocksAggr$Ticker)[1:100]]

y <- StocksAggr[,ReturnQuintile]
x <- StocksAggr[,.SD,.SDcols = featureNames]
xtype <- model.matrix(~StocksAggr$Ticker-1)


x_train = torch_tensor(as.matrix(x), dtype = torch_float())
y_train = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())

# xtype_train = torch_tensor(as.matrix(xtype), dtype = torch_float())

xtype_factor <- as.factor(StocksAggr$Ticker)
i <- torch_tensor(t(cbind(seq_along(xtype_factor),as.integer(xtype_factor))),dtype=torch_int64())
v <- torch_tensor(rep(1,length(xtype_factor)))
xtype_train <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor),length(levels(xtype_factor)) + 10000))
xtype_train <- xtype_train$coalesce()


# layerSizes <- c(length(featureNames), 8, 16, 5)
inputSize <- length(featureNames)
layerSizes <- c(8, 32, 32, 16, 5)
layerTransforms <- c(lapply(1:(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {nnf_softmax(x,2)}))

baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms)
baseModel = prepareBaseModel(baseModel,x = x_train)
criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}

optimizer = optim_adam(baseModel$parameters, lr = 0.01)
epochs = 500
progress <- rep(NA,epochs)
for(i in 1:epochs){
  optimizer$zero_grad()
  y_pred = baseModel(x_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  progress[i] <- loss$item()
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}
plot(progress,type="l")


metaModelInicialized <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1)
metaModel <- metaModelInicialized
optimizer = optim_adam(metaModel$parameters, lr = 0.01)
epochs = 500
progress <- rep(NA,epochs)
start <- Sys.time()
for(i in 1:epochs){
  
  optimizer$zero_grad()
  y_pred = metaModel(x_train,xtype_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  # Check Training
  progress[i] <- loss$item()
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}
plot(progress,type="l")
Sys.time() - start 



mesaModel <- metaModel$MesaModel(metaModel)()
optimizer = optim_adam(mesaModel$parameters, lr = 0.01)
epochs = 300

j <- 1
rowsubset <- xtype_train$indices()[2,] == (j-1)
x_train_subset <- x_train[rowsubset,]
y_train_subset <- y_train[rowsubset]
for(i in 1:epochs){
  
  optimizer$zero_grad()
  y_pred = mesaModel(x_train_subset)
  loss = criterion(y_pred, y_train_subset)
  loss$backward()
  optimizer$step()
  
  # Check Training
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}
t(as.array(metaModel$state_dict()$mesaLayerWeight))[max(1,(j-1)):(j+1)]
mesaModel$state_dict()





# x <- x_train
# xtype <- xtype_train












model <- Model(layerSizes = layerSizes)

state <- model$state_dict()
# state[[str_c("layer_",1,".weight")]]
# state[[str_c("layer_",1,".bias")]]

model(x_train) - model$fforward(x_train, state)





model <- FFNN(x,y)

x_train = torch_tensor(as.matrix(x), dtype = torch_float())
y_train = torch_tensor(t(sapply(y,function(x) replace(numeric(5), x:5, 1))), dtype = torch_float())
y_pred = model(x_train)
ComputeRPSTensor(y_pred,y_train)
ComputeRPSTensor(torch_tensor(matrix(0.2,ncol=5,nrow=length(y)), dtype = torch_float()),y_train)





