library(data.table)
library(stringr)
library(torch)
library(ggplot2)
library(TTR)
rm(list=ls())
tempFilePath <- "C:/Users/stane/M6temp"
source("R/QuantilePrediction/QuantilePrediction_Helpers.R")
source("R/QuantilePrediction/QuantilePrediction_Features.R")
source("R/QuantilePrediction/QuantilePrediction_Models.R")
source("R/MetaModel/MetaModel.R")

StockNames <- readRDS(file.path("Data","StockNames.RDS"))

Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
# Tickers <- StockNames[Symbol %in% names(Stocks),ETF]
# Stocks <- Stocks[Tickers]

# Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
# SP500Tickers <- sample(StockNames[SP500==TRUE,Symbol],100)
# ETFTickers <- sample(StockNames[ETF==TRUE,Symbol],0)
# Stocks <- Stocks[c(SP500Tickers,ETFTickers)]

TimeEnd <- as.Date("2023-01-08")
TimeStart <- TimeEnd - (7*4) * 1000
TimeBreaks <- seq(TimeStart, TimeEnd, b = 7*4) # forecast are made at the break date, ie on x[t]+1 : x[t+1]
TimeBreaks <- TimeBreaks[TimeBreaks>as.Date("1969-12-01")]
TimeBreaksNames <- str_c(TimeBreaks[-length(TimeBreaks)]+1, " : " , TimeBreaks[-1])

# TrainStart <- as.Date("1980-01-07")
TrainStart <- as.Date("2000-01-09")
TrainEnd <- as.Date("2020-01-12")
ValidationStart <- as.Date("2021-12-13")
ValidationEnd <- as.Date("2022-01-09")
# ValidationStart <- as.Date("2022-01-10")
# ValidationEnd <- as.Date("2022-02-06")
# ValidationStart <- as.Date("2022-02-07")
# ValidationEnd <- as.Date("2022-03-06")

featureList <- c(
  list(
    function(SD, BY) {Return(SD)}, #first is just return for y generation
    # function(SD, BY) {LagVolatility(SD, lags = 1:5)},
    function(SD, BY) {LagReturn(SD, lags = 1:5)},
    function(SD, BY) {IsETF(SD, BY, StockNames = StockNames)}
  ),
  TTR
)

# start <- Sys.time()
# s <- 1
# Stocks <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
#   Stock <- Stocks[[s]]
#   Ticker <- names(Stocks)[s]
#   colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
#   Stock <- AugmentStock(Stock[index>=TrainStart & index<=ValidationEnd], ValidationEnd)
#   Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
#   Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
#   Stock[,Ticker := Ticker]
#   Stock
# }))
# StocksAggr <- Stocks[,computeFeatures(.SD, .BY, featureList),.(Ticker)]
# Sys.time() - start 

start <- Sys.time()
StocksAggr <- do.call(rbind,lapply(seq_along(Stocks), function(s) {
  Stock <- Stocks[[s]]
  Ticker <- names(Stocks)[s]
  colnames(Stock) <- c("index", "Open", "High", "Low", "Close", "Volume", "Adjusted")
  Stock <- AugmentStock(Stock[index>=TrainStart & index<=ValidationEnd], ValidationEnd)
  Stock[,Interval := findInterval(index,TimeBreaks,left.open=T)]
  Stock[,Interval := factor(Interval, levels = seq_along(TimeBreaksNames), labels = TimeBreaksNames)]
  Stock[,Ticker := Ticker]
  Stock[,computeFeatures(.SD, .BY, featureList),.(Ticker)]
}))
Sys.time() - start 

featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return"))]
StocksAggr[, ReturnQuintile := computeQuintile(Return), Interval]
StocksAggr[, IntervalStart := as.Date(str_sub(Interval,1,10))]
StocksAggr[, IntervalEnd := as.Date(str_sub(Interval,14,23))]

CheckLeakage <- F
if(CheckLeakage){
  StocksCensored <- Stocks
  StocksCensored <- StocksCensored[index >= ValidationStart, (c("Open", "High", "Low", "Close", "Volume", "Adjusted")) := lapply(1:6,function(x) NA)][]
  StocksCensoredAggr <- StocksCensored[,computeFeatures(.SD, .BY, featureList),.(Ticker)]
  for(featureName in featureNames){
    same <- identical(StocksAggr[, get(featureName)], StocksCensoredAggr[, get(featureName)])
    if(!same){
      stop(str_c("Possible leakage in ", featureName, ", diff=", diff))
    }
  }
}

StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]

TrainRows <- which(StocksAggr[,(IntervalStart >= TrainStart) & (IntervalEnd <= TrainEnd)])
TestRows <- which(StocksAggr[,(IntervalStart > TrainEnd) & (IntervalEnd < ValidationStart)])
ValidationRows <- which(StocksAggr[,(IntervalStart >= ValidationStart) & (IntervalEnd <= ValidationEnd)])

y <- StocksAggr[,ReturnQuintile]
y <- torch_tensor(t(sapply(y,function(x) {
  if(is.na(x)){
    rep(NA,5)
  }else{
    replace(numeric(5), x:5, 1)
  }
})), dtype = torch_float())

x <- StocksAggr[,.SD,.SDcols = featureNames]
x <- torch_tensor(as.matrix(x), dtype = torch_float())

xtype_factor <- as.factor(StocksAggr$Ticker)
i <- torch_tensor(t(cbind(seq_along(xtype_factor),as.integer(xtype_factor))),dtype=torch_int64())
v <- torch_tensor(rep(1,length(xtype_factor)))
xtype <- torch_sparse_coo_tensor(i, v, c(length(xtype_factor),length(levels(xtype_factor))))$coalesce()


y_train <- y[TrainRows,]
x_train <- x[TrainRows,]
xtype_train <- subsetTensor(xtype, rows = TrainRows)
y_test <- y[TestRows,]
x_test <- x[TestRows,]
xtype_test <- subsetTensor(xtype, rows = TestRows)
y_validation <- y[ValidationRows,]
x_validation <- x[ValidationRows,]
xtype_validation <- subsetTensor(xtype, rows = ValidationRows)
criterion = function(y_pred,y) {ComputeRPSTensor(y_pred,y)}


# baseModel ---------------------------------------------------------------

inputSize <- length(featureNames)
layerSizes <- c(32, 8, 5)
# layerSizes <- c(64, 32, 8, 5)
# layerSizes <- c(100, 5)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {nnf_softmax(x,2)}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms)
baseModel = prepareBaseModel(baseModel,x = x_train)
train <- list(y_train, x_train)
test <- list(y_test, x_test)

start <- Sys.time()
fit <- trainModel(model = baseModel, train, test, criterion, epochs = 100, minibatch = Inf, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
Sys.time() - start 
baseModel <- fit$model
baseModelProgress <- fit$progress
y_pred_base <- baseModel(x_validation)
loss_validation_base <- as.array(ComputeRPSTensor(y_pred_base,y_validation))

# metaModel ---------------------------------------------------------------

metaModel <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1)
minibatch <- function() {minibatchSampler(50,xtype_train)}
train <- list(y_train, x_train, xtype_train)
test <- list(y_test, x_test, xtype_test)

start <- Sys.time()
fit <- trainModel(model = metaModel, train, test, criterion, epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 10, printEvery = 1)
Sys.time() - start 
metaModel <- fit$model
metaModelProgress <- fit$progress
y_pred_meta <- metaModel(x_validation, xtype_validation)
loss_validation_meta <- as.array(ComputeRPSTensor(y_pred_meta,y_validation))

# cbind(as.array(y_validation),round(as.array(y_pred),3))

# mesaModel ---------------------------------------------------------------

J <- ncol(xtype_train)
mesaModels <- vector(mode = "list", length = J)
mesaModelsProgress <- vector(mode = "list", length = J)
loss_validation_mesa <- rep(NA,J)

for (j in seq_len(J)){
  if(j %% 10 == 0){
    print(str_c("j: ", j, " Time:", Sys.time()))
  }
  mesaModel <- metaModel$MesaModel(metaModel)()
  rows_train <- xtype_train$indices()[2,] == (j-1)
  x_train_subset <- x_train[rows_train,]
  y_train_subset <- y_train[rows_train,]
  rows_test <- xtype_test$indices()[2,] == (j-1)
  x_test_subset <- x_test[rows_test,]
  y_test_subset <- y_test[rows_test,]
  rows_validation <- xtype_validation$indices()[2,] == (j-1)
  x_validation_subset <- x_validation[rows_validation,]
  y_validation_subset <- y_validation[rows_validation,]
  
  train <- list(y_train_subset, x_train_subset)
  # train <- list(torch_cat(list(y_train_subset, y_test_subset), 1), torch_cat(list(x_train_subset, x_test_subset), 1))
  test <- list(y_test_subset, x_test_subset)
  
  fit <- trainModel(model = mesaModel, train, test, criterion, epochs = 20, minibatch = Inf, tempFilePath = NULL, patience = Inf, printEvery = Inf)
  mesaModel <- fit$model
  mesaModels[[j]] <- mesaModel
  mesaModelsProgress[[j]] <- fit$progress
  y_pred_mesa <- mesaModel(x_validation_subset)
  loss_validation_mesa[j] <- as.array(ComputeRPSTensor(y_pred_mesa,y_validation_subset))
}

temp <- rbind(
  melt(baseModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,type := "base"],
  melt(metaModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,epoch := epoch + baseModelProgress[,which.min(loss_test)]][,type := "meta"]
)
temp_validation <- data.table(
  epoch = c(temp[type=="base",max(epoch)], temp[type=="meta",max(epoch)], temp[type=="meta",max(epoch)]+1),
  variable = "loss_validation",
  value = c(loss_validation_base, loss_validation_meta, mean(loss_validation_mesa)),
  type = c("base", "meta", "mesa")
)
ggplot(temp, aes(x = epoch, y = value, colour =variable, shape=type))+
  geom_line(aes(linetype = type))+
  geom_point(data=temp_validation)+
  geom_text(data=temp_validation, aes(label=round(value,4)), hjust = 0.5, vjust = -1, size=2.2)+
  coord_cartesian(ylim = c(0.145, 0.16))+
  geom_hline(yintercept = 0.16, alpha=0.5)




