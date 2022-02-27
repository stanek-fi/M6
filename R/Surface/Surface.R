# nn_linearAlt <- nn_module(
#   initialize = function(in_features, out_features, bias = T) {
#     initRange <- 1/sqrt(in_features)
#     self$weight = nn_parameter(torch_tensor(matrix(runif(in_features * out_features, -initRange, initRange), nrow = out_features, ncol = in_features)))
#     if(bias){
#       self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
#     }else{
#       self$bias = NULL
#     }
#   },
#   forward = function(input) {
#     nnf_linear(input, weight = self$weight, bias = self$bias)
#   }
# )


fourrier = function(points, theta0, thetac, thetas){
  M <- thetac$size(1)
  N <- thetac$size(2)
  x <- torch_tensor(matrix(seq(0, 2 * pi, length.out = points + 1)[1:points],ncol = points, nrow = M, byrow = T), dtype = torch_float())
  out <- theta0$unsqueeze(2) + Reduce("+",lapply(1:N, function(n) thetac[,n]$unsqueeze(2) * torch_cos(x*n) + thetas[,n]$unsqueeze(2) * torch_sin(x*n)))
  return(out)
}

nn_surfaceFourier <- nn_module(
  initialize = function(in_features, out_features, in_N, out_N, type, bias = T) {
    initRange <- .01
    self$in_features <- in_features
    self$out_features <- out_features
    self$in_N <- in_N
    self$out_N <- out_N
    self$type <- type
    self$includeBias <- bias
    switch (type,
            "CD" = {
              self$theta0 <- nn_parameter(torch_tensor(rep(0, out_features)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
              if(self$includeBias == T){
                self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
              }
            },
            "CC" = {
              if(self$includeBias == T){
                M <- (1 + self$in_N * 2) + 1
              }else{
                M <- (1 + self$in_N * 2)
              }
              self$theta0 <- nn_parameter(torch_tensor(rep(0, M)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
            },
            "DC" = {
              if(self$includeBias == T){
                M <- self$in_features + 1
              }else{
                M <- self$in_features
              }
              self$theta0 <- nn_parameter(torch_tensor(rep(0, M)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(M * out_N, -initRange, initRange), ncol = out_N)))
            }
    )
  },
  forward = function(input) {
    switch (self$type,
            "CD" = {
              weight <- fourrier(self$in_features, self$theta0, self$thetac, self$thetas)
              bias <- self$bias
            },
            "CC" = {
              temp <- fourrier(self$out_features, self$theta0, self$thetac, self$thetas)$transpose(1,2)
              theta0 <- temp[,1]
              thetac <- temp[,(1 + 1):(1 + self$in_N)]
              thetas <- temp[,(1 + self$in_N + 1):(1 + self$in_N + self$in_N)]
              weight <- fourrier(self$in_features, theta0, thetac, thetas)
              if(self$includeBias == T){
                bias <- temp[,(1 + 2 * self$in_N + 1)]
              }else{
                bias <- NULL
              }
            },
            "DC" = {
              temp <- fourrier(self$out_features, self$theta0, self$thetac, self$thetas)$transpose(1,2)
              weight <- temp[,1:self$in_features]
              if(self$includeBias == T){
                bias <- temp[,(self$in_features + 1)]
              }else{
                bias <- NULL
              }
            }
    )
    nnf_linear(input, weight = weight, bias = bias)
  }
)


nn_surfaceBilinear <- nn_module(
  initialize = function(in_features, out_features, in_N, out_N, bias = T) {
    initRange <- 1/sqrt(in_features)
    self$in_features <- in_features
    self$out_features <- out_features
    self$in_N <- min(in_N, in_features)
    self$out_N <- min(out_N, out_features)
    self$includeBias <- bias
    self$weightPar <- nn_parameter(torch_tensor(matrix(runif(self$in_N * self$out_N, -initRange, initRange), ncol=self$in_N, nrow = self$out_N))$unsqueeze(1)$unsqueeze(1))
    if(self$includeBias){
      self$biasPar <- nn_parameter(torch_tensor(matrix(runif(self$out_N, -initRange, initRange), ncol=1, nrow = self$out_N))$unsqueeze(1)$unsqueeze(1))
    }
    # self$constructWeight <- function(weightPar){
    #   
    # }
  },
  forward = function(input) {
    weight <- nnf_interpolate(self$weightPar ,size = c(self$out_features, self$in_features), mode = "bilinear", align_corners = T)[1,1,,]
    if(self$includeBias){
      bias <- nnf_interpolate(self$biasPar ,size = c(self$out_features, 1), mode = "bilinear", align_corners = T)[1,1,,1]
    }else{
      bias <- NULL
    }
    nnf_linear(input, weight = weight, bias = bias)
  }
)