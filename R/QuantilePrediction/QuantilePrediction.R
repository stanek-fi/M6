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


featureList <- c(
  list(
    function(SD, BY) {Return(SD)}, #first is just return for y generation
    function(SD, BY) {LagVolatility(SD, lags = 1:5)},
    function(SD, BY) {LagReturn(SD, lags = 1:5)},
    function(SD, BY) {IsETF(SD, BY, StockNames = StockNames)}
  ),
  TTR
)

# Shifts <- c(0)
Shifts <- c(0,7,14,21)
# Shifts <- c(0,7)
Submission = 0
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)

GenerateStockAggr <- F
if(GenerateStockAggr){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  # temp <- StockNames[M6Dataset>0 & M6Dataset<=2][order(M6Dataset),.(Symbol,M6Dataset)]
  temp <- StockNames[M6Dataset>0][order(M6Dataset),.(Symbol,M6Dataset)]
  Stocks <- Stocks[temp$Symbol]
  M6Datasets <- temp$M6Dataset
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, M6Datasets, CheckLeakage = F)
  saveRDS(StocksAggr, file.path("Precomputed","StocksAggr.RDS"))
}else{
  StocksAggr <- readRDS(file.path("Precomputed","StocksAggr.RDS"))
}

# GenerateStockAggr <- F
# if(GenerateStockAggr){
#   StockNames <- readRDS(file.path("Data","StockNames.RDS"))
#   Stocks <- readRDS(file.path("Data","StocksM6.RDS"))
#   StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, CheckLeakage = F)
#   saveRDS(StocksAggr, file.path("Precomputed","StocksAggr.RDS"))
# }else{
#   StocksAggr <- readRDS(file.path("Precomputed","StocksAggr.RDS"))
# }

# StocksAggr[Interval=="2022-01-17 : 2022-02-13",table(ReturnQuintile),M6Dataset]

featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return", "Shift", "M6Dataset", "ReturnQuintile", "IntervalStart", "IntervalEnd"))]
StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]
# StocksAggr <- StocksAggr[ETF>0]

# TrainStart <- as.Date("1990-01-01")
# TrainStart <- as.Date("1990-01-01")
TrainStart <- as.Date("2010-01-01")
TrainEnd <- as.Date("2019-01-01")
# TrainEnd <- as.Date("2021-01-01")
# ValidationStart <- as.Date("2020-01-01")
# ValidationEnd <- as.Date("2021-01-01")
ValidationStart <- as.Date("2021-01-01")

# ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - 1]
ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - 1]




TrainRows <- which(StocksAggr[,(IntervalStart >= TrainStart) & (IntervalEnd <= TrainEnd)])
TrainRowsM6Dataset <- StocksAggr[TrainRows,M6Dataset]
TestRows <- which(StocksAggr[,(IntervalStart > TrainEnd) & (IntervalEnd < ValidationStart)])
TestRowsM6Dataset <- StocksAggr[TestRows,M6Dataset]
ValidationRows <- which(StocksAggr[,(IntervalStart >= ValidationStart) & (IntervalEnd <= ValidationEnd)])
ValidationRowsM6Dataset <- StocksAggr[ValidationRows,M6Dataset]

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
xtype_factor_M6Dataset <- StocksAggr[,.(M6Dataset = unique(M6Dataset)),Ticker][match(levels(xtype_factor),Ticker)]
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
# layerSizes <- c(32, 5)
layerSizes <- c(32, 8, 5)
# layerSizes <- c(64, 32, 8, 5)
# layerSizes <- c(100, 5)
layerDropouts <- c(rep(0.2, length(layerSizes)-1),0)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_relu), list(function(x) {nnf_softmax(x,2)}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
baseModel = prepareBaseModel(baseModel,x = x_train)
train <- list(y_train, x_train)
test <- list(y_test, x_test)

if(T){
  start <- Sys.time()
  fit <- trainModel(model = baseModel, train, test, criterion, epochs = 500, minibatch = 500, tempFilePath = tempFilePath, patience = 5, printEvery = 1)
  Sys.time() - start 
  baseModel <- fit$model
  baseModelProgress <- fit$progress
  saveRDS(baseModelProgress, file.path("Precomputed","baseModelProgress.RDS"))
  torch_save(baseModel, file.path("Precomputed", str_c("baseModel", ".t7")))
}else{
  baseModelProgress <- readRDS(file.path("Precomputed","baseModelProgress.RDS"))
  baseModel <- torch_load(file.path("Precomputed", str_c("baseModel", ".t7")))
}
y_pred_base <- baseModel(x_validation)
loss_validation_base <- as.array(ComputeRPSTensor(y_pred_base,y_validation))
loss_validation_base_M6Dataset <- sapply(1:max(ValidationRowsM6Dataset), function(i) {
  as.array(ComputeRPSTensor(y_pred_base[which(ValidationRowsM6Dataset == i)], y_validation[which(ValidationRowsM6Dataset == i)]))
})

# lapply(1:max(ValidationRowsM6Dataset), function(i) {
#   temp <- as.array(ComputeRPSTensorVector(y_pred_base[which(ValidationRowsM6Dataset == i)], y_validation[which(ValidationRowsM6Dataset == i)]))
#   # temp <- as.array(ComputeRPSTensorVector(torch_tensor(matrix(0.2,nrow <- length(which(ValidationRowsM6Dataset == i)),ncol=5)), y_validation[which(ValidationRowsM6Dataset == i)]))
#   info <- StocksAggr[ValidationRows][ValidationRowsM6Dataset==i,.(Interval,IntervalStart, IntervalEnd, Ticker, Shift, ETF, Return, ReturnQuintile)]
#   info[,RPS:= temp]
#   # info[,.(mRPS = mean(RPS)),.(Ticker,ETF>0)][order(mRPS)]
#   info[,.(mRPS = mean(RPS),.N),.(ETF>0)][order(ETF)]
# })


# metaModel ---------------------------------------------------------------

metaModel <- MetaModel(baseModel, xtype_train, mesaParameterSize = 1)
minibatch <- function() {minibatchSampler(100,xtype_train)}
train <- list(y_train, x_train, xtype_train)
test <- list(y_test, x_test, xtype_test)

if(T){
  start <- Sys.time()
  fit <- trainModel(model = metaModel, train, test, criterion, epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1)
  Sys.time() - start 
  metaModel <- fit$model
  metaModelProgress <- fit$progress
  saveRDS(metaModelProgress, file.path("Precomputed","metaModelProgress.RDS"))
  torch_save(metaModel, file.path("Precomputed", str_c("metaModel", ".t7")))
}else{
  metaModelProgress <- readRDS(file.path("Precomputed","metaModelProgress.RDS"))
  metaModel <- torch_load(file.path("Precomputed", str_c("metaModel", ".t7")))
}
y_pred_meta <- metaModel(x_validation, xtype_validation)
loss_validation_meta <- as.array(ComputeRPSTensor(y_pred_meta,y_validation))
loss_validation_meta_M6Dataset <- sapply(1:max(ValidationRowsM6Dataset), function(i) {
  as.array(ComputeRPSTensor(y_pred_meta[which(ValidationRowsM6Dataset == i)], y_validation[which(ValidationRowsM6Dataset == i)]))
})


# lapply(1:max(ValidationRowsM6Dataset), function(i) {
#   temp <- as.array(ComputeRPSTensorVector(y_pred_meta[which(ValidationRowsM6Dataset == i)], y_validation[which(ValidationRowsM6Dataset == i)]))
#   info <- StocksAggr[ValidationRows][ValidationRowsM6Dataset==i,.(Interval,IntervalStart, IntervalEnd, Ticker, Shift, ETF, Return, ReturnQuintile)]
#   info[,RPS:= temp]
#   # info[,.(mRPS = mean(RPS)),.(Ticker,ETF>0)][order(mRPS)]
#   info[,.(mRPS = mean(RPS),.N),.(ETF>0)][order(ETF)]
# })


# temp <- as.data.table(cbind(as.array(y_validation),round(as.array(y_pred_meta),3), return=round(StocksAggr[ValidationRows]$Return, 3)))
# View(temp[order(V10+V9-V6-V7)])
# mesaModel ---------------------------------------------------------------

J <- ncol(xtype_train)
mesaModels <- vector(mode = "list", length = J)
mesaModelsProgress <- vector(mode = "list", length = J)
loss_validation_mesa <- rep(NA,J)

for (j in seq_len(J)){
  if(j %% 100 == 0){
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

  # train <- list(y_train_subset, x_train_subset)
  # train <- list(torch_cat(list(y_train_subset, y_test_subset), 1), torch_cat(list(x_train_subset, x_test_subset), 1))
  train <- list(y_test_subset, x_test_subset)
  test <- list(y_test_subset, x_test_subset)

  if(nrow(train[[1]])>0){
    fit <- trainModel(model = mesaModel, train, test, criterion, epochs = 20, minibatch = Inf, tempFilePath = NULL, patience = Inf, printEvery = Inf)
    mesaModel <- fit$model
    mesaModelsProgress[[j]] <- fit$progress
  }
  mesaModels[[j]] <- mesaModel
  y_pred_mesa <- mesaModel(x_validation_subset)
  loss_validation_mesa[j] <- as.array(ComputeRPSTensor(y_pred_mesa,y_validation_subset))
}

temp <- rbind(
  melt(baseModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,type := "base"],
  melt(metaModelProgress[1:which.min(loss_test)],id.vars = "epoch")[,epoch := epoch + baseModelProgress[,which.min(loss_test)]][,type := "meta"]
)
temp_validation <- rbind(
  data.table(
    epoch = c(temp[type=="base",max(epoch)], temp[type=="meta",max(epoch)], temp[type=="meta",max(epoch)]+1),
    variable = "loss_validation",
    value = c(loss_validation_base, loss_validation_meta, mean(loss_validation_mesa)),
    type = c("base", "meta", "mesa"),
    subset = "all"
  ),
  rbind(
    data.table(
      epoch = temp[type=="base",max(epoch)],
      variable = "loss_validation",
      value = loss_validation_base_M6Dataset,
      type = c("base"),
      subset = seq_along(loss_validation_base_M6Dataset)
    ),
    data.table(
      epoch = temp[type=="meta",max(epoch)],
      variable = "loss_validation",
      value = loss_validation_meta_M6Dataset,
      type = c("meta"),
      subset = seq_along(loss_validation_meta_M6Dataset)
    ),
    data.table(
      epoch = temp[type=="meta",max(epoch)]+1,
      variable = "loss_validation",
      value = sapply(seq_along(loss_validation_meta_M6Dataset), function(i) {mean(loss_validation_mesa[xtype_factor_M6Dataset$M6Dataset==i])}),
      type = c("mesa"),
      subset = seq_along(loss_validation_meta_M6Dataset)
    )
  )
)


ggplot(temp, aes(x = epoch, y = value, colour =variable, shape=type))+
  geom_line(aes(linetype = type))+
  geom_point(data=temp_validation[subset=="all"])+
  geom_text(data=temp_validation[subset=="all"], aes(label=round(value,4)), hjust = -0.5, vjust = .2, size=1.9)+
  geom_text(data=temp_validation[subset!="all"], aes(label=subset), hjust = 0.5, vjust = 0.3, size=1.9)+
  geom_text(data=temp_validation[subset=="1"], aes(label=round(value,4)), hjust = -0.5, vjust = .2, size=1.9)+
  coord_cartesian(ylim = c(0.145, 0.16))+
  geom_hline(yintercept = 0.16, alpha=0.5)




# StocksAggrTrain <- StocksAggr[TrainRows, .SD, .SDcols=names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][,Split := "Train"][]
# StocksAggrTest <- StocksAggr[TestRows, .SD, .SDcols=names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][,Split := "Test"][]
# StocksAggrValidation <- StocksAggr[ValidationRows, .SD, .SDcols=names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][,Split := "Validation"][]
# 
# QuantilePredictions <- list(
#   base = rbind(
#     cbind(StocksAggrTrain, setNames(as.data.table(as.array(baseModel(x_train))), str_c("Rank",1:5))),
#     cbind(StocksAggrTest, setNames(as.data.table(as.array(baseModel(x_test))), str_c("Rank",1:5))),
#     cbind(StocksAggrValidation, setNames(as.data.table(as.array(baseModel(x_validation))), str_c("Rank",1:5)))
#   ),
#   meta = rbind(
#     cbind(StocksAggrTrain, setNames(as.data.table(as.array(metaModel(x_train, xtype_train))), str_c("Rank",1:5))),
#     cbind(StocksAggrTest, setNames(as.data.table(as.array(metaModel(x_test, xtype_test))), str_c("Rank",1:5))),
#     cbind(StocksAggrValidation, setNames(as.data.table(as.array(metaModel(x_validation, xtype_validation))), str_c("Rank",1:5)))
#   )
# )
# saveRDS(QuantilePredictions, file.path("Precomputed","QuantilePredictions.RDS"))



# cbind(as.array(y_validation),round(as.array(y_pred),3))
# mesaStates1 <- as.vector(as.array(metaModel$state_dict()$mesaLayerWeight))
# mesaStates2 <- sapply(mesaModels, function(x) as.array(x$state_dict()$mesaState))
# ggplot(data.table(mesaStates1,mesaStates2),aes(x=mesaStates1,mesaStates2))+
#   geom_point()

