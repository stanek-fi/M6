standartize <- function(x){(x - mean(x)) / (sd(x) + 1e-5)}

computeQuintile <- function(x){
  findInterval(rank(x)/length(x),c(0,0.2,0.4,0.6,0.8,1), left.open=T)
} 

ComputeRPSTensor <- function(y_pred,y){
  temp <- (y_pred$cumsum(2) - y)^2
  mean(temp$sum(2)/5)
}

imputeNA <- function(x){
  ifelse(is.na(x),median(x,na.rm = T),x)
}

imputeFeatures <- function(StocksAggr, featureNames = NULL){
  for(featureName in featureNames){
    StocksAggr[[featureName]] <- imputeNA(StocksAggr[[featureName]])
  }
  StocksAggr
}


standartizeFeatures <- function(StocksAggr, featureNames = NULL){
  otherNames <- names(StocksAggr)[!(names(StocksAggr) %in% featureNames)]
  StocksAggr[,c(setNames(lapply(otherNames, function(x) get(x)),otherNames), lapply(.SD, function(x) standartize(x))), Interval, .SDcols = featureNames]
}
