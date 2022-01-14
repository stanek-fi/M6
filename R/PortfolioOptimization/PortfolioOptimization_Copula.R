# library(MASS)
# library(psych)
# library(copula)

rlnormRep <- function(n, mu, sd){
  sd2 <- sd^2
  sdlog <- sqrt(log(sd2 / (exp(2*log(mu))) + 1))
  meanlog <- log(mu) - (sdlog^2)/2
  rlnorm(n = n, meanlog = meanlog, sdlog = sdlog)
}

qlnormRep <- function(p, mu, sd){
  sd2 <- sd^2
  sdlog <- sqrt(log(sd2 / (exp(2*log(mu))) + 1))
  meanlog <- log(mu) - (sdlog^2)/2
  qlnorm(p = p, meanlog = meanlog, sdlog = sdlog)
}

# TObs <- 20
# R <- 2
# N <- 3
# Sigma <-  matrix(c(1, 0.4, 0.2,
#                    0.4, 1, -0.8,
#                    0.2, -0.8, 1),
#                  nrow=3)*2
# # mu <- rep(1,N)
# # sd <- rep(0.025,N)
# mu <- c(1,1.1,0.9)
# sd <- c(0.01,0.02,0.03)

simulateCopula <- function(TObs, R, Sigma, mu, sd = NULL){
  N <- length(mu)
  norm <- sqrt(diag(Sigma))
  if(is.null(sd)){
    sd = norm
  }
  Corr <- diag(1/norm) %*% Sigma %*% diag(1/norm)
  xSim <- array(NA,dim=c(TObs,N,R))
  for(r in seq_len(R)){
    z <- mvrnorm(TObs, mu=rep(0, N), Sigma=Corr, empirical=F)
    u <- pnorm(z)
    
    x <- do.call(cbind,lapply(seq_len(ncol(u)), function(i) {
      qlnormRep(u[,i],mu[i],sd[i])
    }))
    xSim[,,r] <- x
  }
  return(xSim)
}

# x <- simulateCopula(10000,1,Sigma,mu,sd)
# apply(x[,,1],2,mean)
# apply(x[,,1],2,sd)
# cor(x[,,1])
# pairs.panels(x)


















