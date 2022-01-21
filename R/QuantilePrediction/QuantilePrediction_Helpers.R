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
  # if(is.numeric(x)){
  #   ifelse(is.na(x),median(x),x)
  # }else{
  #   x
  # }
}