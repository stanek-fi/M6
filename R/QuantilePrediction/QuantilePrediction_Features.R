computeFeatures <- function(SD,featureList){
  temp <- lapply(featureList, function(f) f(SD))
  control <- lapply(temp, function(x) x[,Interval])
  if(length(unique.default(control))>1){
    stop("Some feature does not align.")
  }
  cbind(Interval = control[[1]], do.call(cbind,lapply(temp, function(x) x[,-1])))
}

# SD <- Stocks[Ticker == "ABBV"]

firstNum <- function(x){
  temp <- na.omit(x)
  if(length(temp)>0){
    first(temp)
  }else{
    NA
  }
}

lastNum <- function(x){
  temp <- na.omit(x)
  if(length(temp)>0){
    last(temp)
  }else{
    NA
  }
}

Return <- function(SD) {
  temp <- SD[,.(Return = lastNum(Adjusted)/firstNum(Adjusted) - 1), Interval]
  return(temp)
}

LagReturn <- function(SD,lags = 1) {
  temp <- SD[,.(Return = lastNum(Adjusted)/firstNum(Adjusted) - 1), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Return, n=lags, fill = NA)]))
  names(temp) = c("Interval", str_c("ReturnLag",lags))
  return(temp)
}

LagVolatility <- function(SD,lags = 1) {
  temp <- SD[,.(Volatility = mean(diff(log(na.omit(Adjusted)))^2)), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Volatility, n=lags, fill = NA)]))
  names(temp) = c("Interval", str_c("VolatilityLag",lags))
  return(temp)
}


TTRWrapper <- function(SD, f, SDcols, Normalize, SDcolsOut = NULL, SDcolsPlus = NULL, Transform = NULL, ...){
  naRows <- apply(is.na(SD[,.SD,.SDcols=SDcols]),1,any)
  if(is.null(SDcolsPlus)){
    temp <- f(SD[!naRows,.SD,.SDcols=SDcols], ...)
  }else{
    temp <- f(SD[!naRows,.SD,.SDcols=SDcols], SD[!naRows,.SD,.SDcols=SDcolsPlus], ...)
  }
  if(is.vector(temp)){
    temp <- as.data.frame(temp)
    if(!is.null(SDcolsOut)){
      colnames(temp)=SDcolsOut
    }
  }
  temp <- temp[ifelse(naRows,NA,cumsum(!naRows)),,drop=F]
  SDcolsOut <- colnames(temp)
  if(length(Normalize)==1){
    Normalize <- rep(Normalize, length(SDcolsOut))
  }
  for(i in seq_along(Normalize)){
    if(Normalize[i]){
      temp[,i] <- temp[,i] / SD$Close
    }
  }
  if(!is.null(Transform)){
    if(length(Transform)==1){
      Transform <- rep(Transform, length(SDcolsOut))
    }
    for(i in seq_along(Transform)){
      temp[,i] <- Transform[[i]](temp[,i]) 
    }
  }
  temp <- cbind(SD,temp)
  temp <- temp[,lapply(.SD, function(x) mean(x, na.rm = T)),Interval, .SDcols=SDcolsOut]
  temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=SDcolsOut]
}


TTR <- list(
  # function(SD) {TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = F)},
  # function(SD) {TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F)},
  # function(SD) {TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T)},
  # function(SD) {TTRWrapper(SD = SD, f = BBands, SDcols = c("High", "Low", "Close"), Normalize = c(T, T, T, F))},
  # function(SD) {TTRWrapper(SD = SD, f = CCI, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "cci")},
  # function(SD) {TTRWrapper(SD = SD, f = chaikinAD, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "chaikinAD", SDcolsPlus = "Volume", Transform = list(function(x) c(NA,diff(x))))},
  # function(SD) {TTRWrapper(SD = SD, f = chaikinVolatility, SDcols = c("High", "Low"),Normalize = F, SDcolsOut = "chaikinVolatility")},
  # function(SD) {TTRWrapper(SD = SD, f = CLV, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsOut = "CLV")},
  # function(SD) {TTRWrapper(SD = SD, f = CMF, SDcols = c("High", "Low", "Close"), Normalize = F, SDcolsPlus = "Volume", SDcolsOut = "CMF")},
  # function(SD) {TTRWrapper(SD = SD, f = CMO, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CMO")},
  function(SD) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI")} #this one migh be slow
  # function(SD) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI", n=5)},
  # function(SD) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI", n=10)},
  # function(SD) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI", n=20)},
  # function(SD) {TTRWrapper(SD = SD, f = CTI, SDcols = c("Adjusted"), Normalize = F, SDcolsOut = "CTI", n=40)}
)

# 
# TTR_ADX <- function(SD) {
#   TTRWrapper(SD = SD, f = ADX, SDcols = c("High", "Low", "Close"), Normalize = F)
# }
# 
# TTR_aroon <- function(SD) {
#   TTRWrapper(SD = SD, f = aroon, SDcols = c("High", "Low"), Normalize = F)
# }
# 
# TTR_ATR <- function(SD){
#   TTRWrapper(SD = SD, f = ATR, SDcols = c("High", "Low", "Close"), Normalize = T)
# }
# 





