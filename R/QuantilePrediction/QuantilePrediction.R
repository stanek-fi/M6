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
    function(SD, BY) {LagVolatility(SD, lags = 1:7)},
    function(SD, BY) {LagReturn(SD, lags = 1:7)},
    function(SD, BY) {IsETF(SD, BY, StockNames = StockNames)}
  ),
  TTR
)

# Shifts <- c(0)
Shifts <- c(0,7,14,21)
# Shifts <- c(0,7)
Submission = 10
IntervalInfos <- GenIntervalInfos(Submission = Submission, Shifts = Shifts)

GenerateStockAggr <- F
if(GenerateStockAggr){
  StockNames <- readRDS(file.path("Data","StockNames.RDS"))
  StockNames[Symbol=="FB",Symbol := "META"]  # * needed to account for the fact that one cannot download FB data under FB ticker anymore
  StockNames <- StockNames[Symbol!="ANAT"]  # * not quite sure why I need to drop this
  Stocks <- readRDS(file.path("Data","StocksAll.RDS"))
  # temp <- StockNames[M6Dataset>0 & M6Dataset<=2][order(M6Dataset),.(Symbol,M6Dataset)]
  # temp <- StockNames[M6Dataset>0][order(M6Dataset),.(Symbol,M6Dataset)]
  # temp <- StockNames[M6Dataset>0 & Symbol != "JW-A"][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  # temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  # temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS", "ANAT", "ENIA"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  # temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS", "ANAT", "ENIA", "ACC"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  temp <- StockNames[M6Dataset>0 & !(Symbol %in% c("JW-A", "NCBS", "ANAT", "ENIA", "ACC", "LFC", "SHI"))][order(M6Dataset),.(Symbol,M6Dataset)] #TODO: this is temp fix fore exluding the  JW-A stock which could not be downloaded. Update stock names to get this fixed
  Stocks <- Stocks[temp$Symbol]
  M6Datasets <- temp$M6Dataset
  StocksAggr <- GenStocksAggr(Stocks, IntervalInfos, featureList, M6Datasets, CheckLeakage = F)
  # StocksAggr <- GenStocksAggr(Stocks[800:1000], IntervalInfos, featureList, M6Datasets[800:1000], CheckLeakage = F)
  saveRDS(StocksAggr, file.path("Precomputed","StocksAggr.RDS"))
}else{
  StocksAggr <- readRDS(file.path("Precomputed","StocksAggr.RDS"))
}

# StocksAggr[Shif@t == 0 & M6Dataset == 1]

featureNames <- names(StocksAggr)[!(names(StocksAggr) %in% c("Ticker", "Interval", "Return", "Shift", "M6Dataset", "ReturnQuintile", "IntervalStart", "IntervalEnd"))]
StocksAggr <- imputeFeatures(StocksAggr, featureNames = featureNames)
# StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames)
StocksAggr <- standartizeFeatures(StocksAggr, featureNames = featureNames[!(featureNames %in% c("ETF"))])
StocksAggr <- StocksAggr[order(IntervalStart,Ticker)]



TrainStart <- as.Date("2000-01-01")
# TrainStart <- as.Date("2010-01-01")
# TrainEnd <- as.Date("2020-01-01")
TrainEnd <- as.Date("2022-01-01")
# TrainEnd <- as.Date("2022-03-01")
# ValidationStart <- as.Date("2021-01-01")
# ValidationEnd <- as.Date("2022-01-01")
q <- 0
ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - q]
ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - q]


# TrainStart <- as.Date("2000-01-01")
# TrainEnd <- as.Date("2020-01-12")
# q <- 1
# ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - q]
# ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - q]

# TrainStart <- as.Date("2000-01-01")
# TrainEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - 28]
# q <- 0
# ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - q]
# ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - q]

# TrainStart <- as.Date("2000-01-01")
# TrainEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission) - (28-12)]
# ValidationStart <- IntervalInfos[[1]]$IntervalStarts[length(IntervalInfos[[1]]$IntervalStarts) - (12 - Submission) - (12-6)]
# ValidationEnd <- IntervalInfos[[1]]$IntervalEnds[length(IntervalInfos[[1]]$IntervalEnds) - (12 - Submission)]



TrainRows <- which(StocksAggr[,(IntervalStart >= TrainStart) & (IntervalEnd <= TrainEnd)])
# TrainRowsM6Dataset <- StocksAggr[TrainRows,M6Dataset]
TrainInfo <- StocksAggr[TrainRows,.(Interval, IntervalStart, IntervalEnd, Shift, M6Dataset, Ticker, Return)]
TestRows <- which(StocksAggr[,(IntervalStart > TrainEnd) & (IntervalEnd < ValidationStart)])
# TestRowsM6Dataset <- StocksAggr[TestRows,M6Dataset]
TestInfo <- StocksAggr[TestRows,.(Interval, IntervalStart, IntervalEnd, Shift, M6Dataset, Ticker, Return)]
ValidationRows <- which(StocksAggr[,(IntervalStart >= ValidationStart) & (IntervalEnd <= ValidationEnd)])
ValidationInfo <- StocksAggr[ValidationRows,.(Interval, IntervalStart, IntervalEnd, Shift, M6Dataset, Ticker, Return)]

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
r <- 1
set.seed(r)
torch_manual_seed(r)

inputSize <- length(featureNames)
layerSizes <- c(32, 8, 5)
layerDropouts <- c(rep(0.2, length(layerSizes)-1),0)
layerTransforms <- c(lapply(seq_len(length(layerSizes)-1), function(x) nnf_leaky_relu), list(function(x) {nnf_softmax(x,2)}))
baseModel <- constructFFNN(inputSize, layerSizes, layerTransforms, layerDropouts)
baseModel = prepareBaseModel(baseModel,x = x_train)
# train <- list(y_train, x_train)
# test <- list(y_test, x_test)
# validation <- list(y_validation, x_validation)
# minibatch <- 1000
minibatch <- 200

# lr <- c(0.01,0.001,0.0005,0.0001)
lr <- c(0.01)

if(T){
  start <- Sys.time()
  # fit <- trainModel(model = baseModel, criterion, train, test, validation, epochs = 100, minibatch = 1000, tempFilePath = tempFilePath, patience = 5, printEvery = 1)
  # fit <- trainModel(model = baseModel, criterion, train = list(y_train, x_train), test = list(y_test, x_test), validation = list(y_validation, x_validation), epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr=lr)
  fit <- trainModel2(model = baseModel, criterion, train = list(y_train, x_train), test = list(y_test, x_test), validation = list(y_validation, x_validation), epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr=lr)
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
loss_validation_base_vector <- as.array(ComputeRPSTensorVector(y_pred_base,y_validation))
loss_validation_base_M6Dataset <- sapply(1:max(ValidationInfo$M6Dataset), function(i) {mean(loss_validation_base_vector[which(ValidationInfo$M6Dataset == i)])})


# temp <- cbind(ValidationInfo,RPS = loss_validation_base_vector)
# # # temp <- cbind(TestInfo,RPS = as.array(ComputeRPSTensorVector(baseModel(x_test),y_test)))
# temp <- temp[,.(RPS = mean(RPS), MV = mean(Return^2)), .(Shift, IntervalStart, M6Dataset)]
# ggplot(temp, aes(x=IntervalStart,y=RPS,colour=as.factor(M6Dataset)))+
#   geom_line()+geom_point()+
#   # geom_line(aes(y=MV * 14))+
#   facet_grid(Shift~.)

# metaModel ---------------------------------------------------------------

# X_MtMsOLS <- as.matrix(x_train)[,1:26]
# y_MtMsOLS <- as.matrix(StocksAggr[TrainRows,Return])^2
# group_MtMsOLS <- as.vector(as.matrix(xtype_train$indices()[2,])+1)
# fit_MtMsOLS <- MtMsOLS(y_MtMsOLS, X_MtMsOLS, group_MtMsOLS, s = 2, lambda = 0, R = 40, prefit = T, printEvery = 1)

horizon <- NULL
r <- 1
set.seed(r)
torch_manual_seed(r)


allowMetaStructure <- rapply(baseModel$stateStructure, function(x) torch_tensor(array(F, dim = x)), how = "list")
includedLayers <- (length(baseModel$stateStructure)-1):length(baseModel$stateStructure)
print(baseModel$stateStructure[includedLayers])
for (i in includedLayers) {
    allowMetaStructure[[i]] <- torch_tensor(array(T, dim = dim(allowMetaStructure[[i]])))
} 

# metaModel <- MetaModel(
#   baseModel, 
#   xtype_train, 
#   mesaParameterSize = 2, 
#   allowBias = F, 
#   pDropout = 0.1,  
#   initMesaRange = 0, 
#   initMetaRange = 0.7
# )

metaModel <- MetaModel(
  baseModel, 
  xtype_train, 
  mesaParameterSize = 1, 
  allowBias = T, 
  pDropout = 0,  
  initMesaRange = 0, 
  initMetaRange = 1,
  allowMetaStructure = allowMetaStructure
)
# state <- metaModel$state_dict()
# temp <- fit_MtMsOLS$theta_s
# temp[is.na(temp)]=0
# temp <- temp/sd(temp)/2
# state$mesaLayerWeight <- torch_tensor(t(temp), requires_grad = T)
# state$metaLayerWeight <- torch_tensor(matrix(0, nrow = nrow(state$metaLayerWeight), ncol = ncol(state$metaLayerWeight)), requires_grad = T)
# metaModel$load_state_dict(state)

minibatch <- function() {minibatchSampler(100,xtype_train)}
# minibatch <- function() {minibatchSampler(5,xtype_train)}
# minibatch <- 10000
# lr <- c(0.01,0.001,0.001,0.0005,0.0003,0.0001,0.00005)
lr <- c(0.01,0.001,0.0005,0.0001)
# lr <- c(0.001)
# train <- list(y_train, x_train, xtype_train)
# rows <- StocksAggr[TrainRows][,which(IntervalStart > as.Date("2010-01-10"))]
# train <- list(subsetTensor(y_train,rows), subsetTensor(x_train,rows), subsetTensor(xtype_train,rows))
# minibatch <- function() {minibatchSampler(5,train[[3]])}
# test <- list(y_test, x_test, xtype_test)
# validation <- list(y_validation, x_validation, xtype_validation)

if(T){
  start <- Sys.time()
  # fit <- trainModel(model = metaModel, criterion, train = list(y_train, x_train, xtype_train), test = list(y_test, x_test, xtype_test), validation = list(y_validation, x_validation, xtype_validation), epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr)
  # fit <- trainModel(model = metaModel, criterion, train = train, test = list(y_test, x_test, xtype_test), validation = list(y_validation, x_validation, xtype_validation), epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 5, printEvery = 1, lr = lr)
  fit <- trainModel2(model = metaModel, criterion, train = list(y_train, x_train, xtype_train), test = list(y_test, x_test, xtype_test), validation = list(y_validation, x_validation, xtype_validation), epochs = 100, minibatch = minibatch, tempFilePath = tempFilePath, patience = 10, printEvery = 1, lr = lr)
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
loss_validation_meta_vector <- as.array(ComputeRPSTensorVector(y_pred_meta,y_validation))
loss_validation_meta_M6Dataset <- sapply(1:max(ValidationInfo$M6Dataset), function(i) {mean(loss_validation_meta_vector[which(ValidationInfo$M6Dataset == i)])})


# mesaModel ---------------------------------------------------------------

J <- ncol(xtype_train)
mesaModels <- vector(mode = "list", length = J)
mesaModelsProgress <- vector(mode = "list", length = J)
loss_validation_mesa <- rep(NA,J)

# for (j in seq_len(J)){
#   if(j %% 100 == 0){
#     print(str_c("j: ", j, " Time:", Sys.time()))
#   }
#   mesaModel <- metaModel$MesaModel(metaModel)()
#   rows_train <- xtype_train$indices()[2,] == (j-1)
#   x_train_subset <- x_train[rows_train,]
#   y_train_subset <- y_train[rows_train,]
#   rows_test <- xtype_test$indices()[2,] == (j-1)
#   x_test_subset <- x_test[rows_test,]
#   y_test_subset <- y_test[rows_test,]
#   rows_validation <- xtype_validation$indices()[2,] == (j-1)
#   x_validation_subset <- x_validation[rows_validation,]
#   y_validation_subset <- y_validation[rows_validation,]
#   train <- list(torch_cat(list(y_train_subset, y_test_subset), 1), torch_cat(list(x_train_subset, x_test_subset), 1))
#   # train <- list(y_test_subset, x_test_subset)
#   # test <- list(y_test_subset, x_test_subset)
#   if(nrow(train[[1]])>0){
#     fit <- trainModel(model = mesaModel, criterion, train, epochs = 10, minibatch = Inf, tempFilePath = NULL, patience = Inf, printEvery = Inf)
#     mesaModel <- fit$model
#     mesaModelsProgress[[j]] <- fit$progress
#   }
#   mesaModels[[j]] <- mesaModel
#   y_pred_mesa <- mesaModel(x_validation_subset)
#   loss_validation_mesa[j] <- as.array(ComputeRPSTensor(y_pred_mesa,y_validation_subset))
# }

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




StocksAggrTrain <- StocksAggr[TrainRows, .SD, .SDcols=names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][,Split := "Train"][]
StocksAggrTest <- StocksAggr[TestRows, .SD, .SDcols=names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][,Split := "Test"][]
StocksAggrValidation <- StocksAggr[ValidationRows, .SD, .SDcols=names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]][,Split := "Validation"][]

QuantilePredictions <- list(
  base = rbind(
    cbind(StocksAggrTrain, setNames(as.data.table(as.array(baseModel(x_train))), str_c("Rank",1:5))),
    cbind(StocksAggrTest, setNames(as.data.table(as.array(baseModel(x_test))), str_c("Rank",1:5))),
    cbind(StocksAggrValidation, setNames(as.data.table(as.array(baseModel(x_validation))), str_c("Rank",1:5)))
  ),
  meta = rbind(
    cbind(StocksAggrTrain, setNames(as.data.table(as.array(metaModel(x_train, xtype_train))), str_c("Rank",1:5))),
    cbind(StocksAggrTest, setNames(as.data.table(as.array(metaModel(x_test, xtype_test))), str_c("Rank",1:5))),
    cbind(StocksAggrValidation, setNames(as.data.table(as.array(metaModel(x_validation, xtype_validation))), str_c("Rank",1:5)))
  )
)
saveRDS(QuantilePredictions, file.path("Precomputed","QuantilePredictions.RDS"))



# cbind(as.array(y_validation),round(as.array(y_pred),3))
# mesaStates1 <- as.vector(as.array(metaModel$state_dict()$mesaLayerWeight))
# mesaStates2 <- sapply(mesaModels, function(x) as.array(x$state_dict()$mesaState))
# ggplot(data.table(mesaStates1,mesaStates2),aes(x=mesaStates1,mesaStates2))+
#   geom_point()

