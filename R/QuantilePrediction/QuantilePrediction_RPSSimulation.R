N <- 100
k <- 5

ComputeRPSMatrix <- function(y_pred,y){
  temp <- (t(apply(y_pred,1,cumsum)) - y)^2
  mean(apply(temp,1,mean))
}

R <- 100
res <- rep(NA,R)
for(r in 1:R){
  y <- sample(rep(1:k,each=N/k))
  y <- t(sapply(y,function(x) replace(numeric(k), x:k, 1)))
  
  # y_pred <- matrix(1/k,nrow=N,ncol=k)
  
  # y_pred <- matrix(0,nrow=N,ncol=k)
  # y_pred[,3]=1
  
  alpha <- 0.01
  y_pred <- matrix(1/k,nrow=N,ncol=k)
  y_pred <- y_pred-alpha
  y_pred[,2:4] <- y_pred[,2:4] + alpha*k/3
  
  res[r] <- ComputeRPSMatrix(y_pred,y)
}
mean(res)
hist(res)





k <- 2
N <- 10
R <- 10000
res <- rep(NA,R)
treshold <- 0.6
for(r in 1:R){
  
  if(runif(1) < treshold ){
    q <- c(1,2)
  }else{
    q <- c(2,1)
  }
  y <- rep(q,each=N/k)
  y <- t(sapply(y,function(x) replace(numeric(k), x:k, 1)))
  
  # y_pred <- matrix(1/k,nrow=N,ncol=k)
  
  # y_pred <- matrix(0,nrow=N,ncol=k)
  # y_pred[,1]=1
  
  alpha <- 0.9
  y_pred <- rbind(
    matrix(c(treshold*alpha,1-treshold*alpha),nrow = N/2, ncol=k,byrow = T),
    matrix(c(1-treshold*alpha,treshold*alpha),nrow = N/2, ncol=k,byrow = T)
  )

  res[r] <- ComputeRPSMatrix(y_pred,y)
}
mean(res)
hist(res)




