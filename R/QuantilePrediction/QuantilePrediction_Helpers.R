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


# FFNNSoftmax = nn_module(
#   initialize = function(layerSizes) {
#     self$layerSizes <- layerSizes
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       self[[str_c("layer_",i)]] <- nn_linear(layerSizes[i], layerSizes[i+1])
#     }
#   },
#   forward = function(x) {
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- nnf_relu(self[[str_c("layer_",i)]](x))
#     }
#     nnf_softmax(x,2)
#   },
#   fforward = function(x,state){
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- nnf_relu(nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
#     }
#     nnf_softmax(x,2)
#   }
# )

  
constructFFNN = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_linear(self$layerSizesAll[i], self$layerSizesAll[i+1])
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
    }
    x
  },
  fforward = function(x,state){
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]]))
    }
    x
  }
)


# FFNNSoftmax = nn_module(
#   initialize = function(layerSizes) {
#     self$layerSizes <- layerSizes
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       self[[str_c("layer_",i)]] <- nn_linear(layerSizes[i], layerSizes[i+1])
#     }
#   },
#   forward = function(x) {
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- self[[str_c("layer_",i)]](x)
#     }
#     nnf_softmax(x,2)
#   },
#   fforward = function(x,state){
#     for(i in seq_along(self$layerSizes)[-length(self$layerSizes)]){
#       x <- nnf_linear(x, weight = state[[str_c("layer_",i,".weight")]], bias = state[[str_c("layer_",i,".bias")]])
#     }
#     nnf_softmax(x,2)
#   }
# )

