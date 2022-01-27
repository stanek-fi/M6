computeFeatures <- function(SD,featureList){
  temp <- lapply(featureList, function(f) f(SD))
  control <- lapply(temp, function(x) x[,Interval])
  if(length(unique.default(control))>1){
    stop("Some feature does not align.")
  }
  cbind(Interval = control[[1]], do.call(cbind,lapply(temp, function(x) x[,-1])))
}

# SD <- Stocks[Ticker == "AVB"]

Return <- function(SD) {
  temp <- SD[,.(Return = last(Adjusted)/first(Adjusted) - 1), Interval]
  return(temp)
}

LagReturn <- function(SD,lags = 1) {
  temp <- SD[,.(Return = last(Adjusted)/first(Adjusted) - 1), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Return, n=lags, fill = NA)]))
  names(temp) = c("Interval", str_c("ReturnLag",lags))
  return(temp)
}
LagVolatility <- function(SD,lags = 1) {
  temp <- SD[,.(Volatility = mean(diff(log(Adjusted))^2)), Interval]
  temp <- cbind(temp[,.(Interval)], as.data.table(temp[,shift(Volatility, n=lags, fill = NA)]))
  names(temp) = c("Interval", str_c("VolatilityLag",lags))
  return(temp)
}










