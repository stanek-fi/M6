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

TTR_ADX <- function(SD) {
  SDcols <- c("High", "Low", "Close")
  naRows <- apply(is.na(SD[,.SD,.SDcols=SDcols]),1,any)
  temp <- ADX(SD[!naRows,.SD,.SDcols=SDcols])
  temp <- temp[ifelse(naRows,NA,cumsum(!naRows)),]
  SDcols <- colnames(temp)
  temp <- cbind(SD,temp)
  temp <- temp[,lapply(.SD, function(x) mean(x, na.rm = T)),Interval, .SDcols=SDcols]
  temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=SDcols]
  return(temp)
}

TTR_aroon <- function(SD) {
  SDcols <- c("High", "Low")
  naRows <- apply(is.na(SD[,.SD,.SDcols=SDcols]),1,any)
  temp <- aroon(SD[!naRows,.SD,.SDcols=SDcols])
  temp <- temp[ifelse(naRows,NA,cumsum(!naRows)),]
  SDcols <- colnames(temp)
  temp <- cbind(SD,temp)
  temp <- temp[,lapply(.SD, function(x) mean(x, na.rm = T)),Interval, .SDcols=SDcols]
  temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=SDcols]
  return(temp)
}

# TTR_aroon_old <- function(SD) {
#   temp <- cbind(SD,aroon(SD[,.(High,Low)]))
#   temp <- temp[,lapply(.SD, function(x) mean(x)),Interval, .SDcols=c("aroonUp", "aroonDn", "oscillator")]
#   temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=c("aroonUp", "aroonDn", "oscillator")]
#   return(temp)
# }
# 
# TTR_aroonLast <- function(SD) {
#   temp <- cbind(SD,aroon(SD[,.(High,Low)]))
#   temp <- temp[,lapply(.SD, function(x) last(x)),Interval, .SDcols=c("aroonUp", "aroonDn", "oscillator")]
#   temp <- temp[,c(.(Interval = Interval),lapply(.SD, function(x) shift(x, n=1, fill = NA))), .SDcols=c("aroonUp", "aroonDn", "oscillator")]
#   return(temp)
# }








