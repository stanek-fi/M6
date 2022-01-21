noisyInterpolation <- function(x){
  # na_interpolation(x) + ifelse(is.na(x), rnorm(length(x), 0, sd(diff(x),na.rm = T)), 0)
  whichna <- which(is.na(x))
  intervals <- split(whichna, cumsum(c(1, diff(whichna) != 1)))
  xnoise <- rep(0,length(x))
  xomited <- na.omit(x)
  for (i in seq_along(intervals)){
    interval <- intervals[[i]]
    if(length(interval)<= length(xomited)){
      startindex <- sample(1:(length(xomited)-length(interval) + 1),1)
      xsub <- xomited[startindex:(startindex + length(interval) - 1)]
      xsub <- xsub - xsub[1]
      xsub <- xsub - (seq_along(xsub)-1)/(length(xsub)) * xsub[length(xsub)]
    }else{
      xsub <- rep(0,length(interval))
    }
    xnoise[interval] = xsub
  }
  na_interpolation(x) + xnoise
}