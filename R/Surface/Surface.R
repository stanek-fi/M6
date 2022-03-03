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


fourier = function(points, theta0, thetac, thetas){
  M <- thetac$size(1)
  N <- thetac$size(2)
  x <- torch_tensor(matrix(seq(0, 2 * pi, length.out = points + 1)[1:points],ncol = points, nrow = M, byrow = T), dtype = torch_float())
  out <- theta0$unsqueeze(2) + Reduce("+",lapply(1:N, function(n) thetac[,n]$unsqueeze(2) * torch_cos(x*n) + thetas[,n]$unsqueeze(2) * torch_sin(x*n)))
  return(out)
}

nn_surfaceFourier <- nn_module(
  initialize = function(in_features, out_features, in_N, out_N, bias = T) {
    initRange <- .01
    self$in_features <- in_features
    self$out_features <- out_features
    self$in_N <- min(in_N, in_features)
    self$out_N <- min(out_N, out_features)
    if((self$in_N < in_features) & (self$out_N < out_features)){
      self$type <- "CC"
    } else if (self$in_N == in_features){
      self$type <- "DC"
    } else if (self$out_N == out_features){
      self$type <- "CD"
    }
    self$includeBias <- bias
    switch (self$type,
            "CD" = {
              self$theta0 <- nn_parameter(torch_tensor(rep(0, out_features)))
              self$thetac <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
              self$thetas <- nn_parameter(torch_tensor(matrix(runif(in_N * out_features, -initRange, initRange), ncol = in_N)))
              if(self$includeBias == T){
                self$bias = nn_parameter(torch_tensor(runif(out_features, -initRange, initRange)))
              }
              self$constructWB<- function(theta0, thetac, thetas, bias = NULL){
                list(
                  weight = fourier(self$in_features, theta0, thetac, thetas),
                  bias = bias
                )
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
              self$constructWB <- function(theta0, thetac, thetas, bias = NULL){
                temp <- fourier(self$out_features, theta0, thetac, thetas)$transpose(1,2)
                theta0 <- temp[,1]
                thetac <- temp[,(1 + 1):(1 + self$in_N)]
                thetas <- temp[,(1 + self$in_N + 1):(1 + self$in_N + self$in_N)]
                weight <- fourier(self$in_features, theta0, thetac, thetas)
                if(self$includeBias == T){
                  bias <- temp[,(1 + 2 * self$in_N + 1)]
                }else{
                  bias <- NULL
                }
                list(
                  weight = weight,
                  bias = bias
                )
              }
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
              self$constructWB <- function(theta0, thetac, thetas, bias = NULL){
                temp <- fourier(self$out_features, theta0, thetac, thetas)$transpose(1,2)
                weight <- temp[,1:self$in_features]
                if(self$includeBias == T){
                  bias <- temp[,(self$in_features + 1)]
                }else{
                  bias <- NULL
                }
                list(
                  weight = weight,
                  bias = bias
                )
              }
            }
    )
  },
  forward = function(input) {
    temp <- self$constructWB(self$theta0, self$thetac, self$thetas, self$bias)
    nnf_linear(input, weight = temp$weight, bias = temp$bias)
  }
)

constructFFNSF = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, in_Ns, out_Ns, layerDropouts = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_surfaceFourier(self$layerSizesAll[i], self$layerSizesAll[i+1], in_N = in_Ns[i], out_N = out_Ns[i])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
    }
    x
  },
  fforward = function(x,state){
    for(i in seq_along(self$layerSizes)){
      temp <- self[[str_c("layer_",i)]]$constructWB(
        state[[str_c("layer_",i,".theta0")]], 
        state[[str_c("layer_",i,".thetac")]], 
        state[[str_c("layer_",i,".thetas")]], 
        state[[str_c("layer_",i,".bias")]]
      )
      x <- self$layerTransforms[[i]](
        nnf_linear(x, weight = temp$weight, bias = temp$bias)
      )
    }
    x
  }
)



nn_surfaceBilinear <- nn_module(
  initialize = function(in_features, out_features, in_N, out_N, bias = T) {
    initRange <- 1/sqrt(in_features)
    self$in_features <- in_features
    self$out_features <- out_features
    self$in_N <- min(in_N, in_features)
    self$out_N <- min(out_N, out_features)
    # self$includeBias <- bias
    self$weightPar <- nn_parameter(torch_tensor(matrix(runif(self$in_N * self$out_N, -initRange, initRange), ncol=self$in_N, nrow = self$out_N))$unsqueeze(1)$unsqueeze(1))
    if(bias){
      self$biasPar <- nn_parameter(torch_tensor(matrix(runif(self$out_N, -initRange, initRange), ncol=1, nrow = self$out_N))$unsqueeze(1)$unsqueeze(1))
    }
    self$constructWeight <- function(weightPar){
      nnf_interpolate(weightPar ,size = c(self$out_features, self$in_features), mode = "bilinear", align_corners = T)[1,1,,]
    }
    self$constructBias <- function(biasPar){
      if(!is.null(biasPar)){
        nnf_interpolate(biasPar ,size = c(self$out_features, 1), mode = "bilinear", align_corners = T)[1,1,,1]
      }else{
        NULL
      }
    }
  },
  forward = function(input) {
    weight <- self$constructWeight(self$weightPar)
    bias <- self$constructBias(self$biasPar)
    nnf_linear(input, weight = weight, bias = bias)
  }
)

constructFFNSB = nn_module(
  initialize = function(inputSize, layerSizes, layerTransforms, in_Ns, out_Ns, layerDropouts = NULL) {
    self$layerSizes <- layerSizes
    self$layerTransforms <- layerTransforms
    self$layerSizesAll <- c(inputSize, layerSizes)
    self$Dropout <- !is.null(layerDropouts)
    for(i in seq_along(self$layerSizes)){
      self[[str_c("layer_",i)]] <- nn_surfaceBilinear(self$layerSizesAll[i], self$layerSizesAll[i+1], in_N = in_Ns[i], out_N = out_Ns[i])
    }
    if(self$Dropout){
      for(i in seq_along(self$layerSizes)){
        self[[str_c("layerDropout_",i)]] <- nn_dropout(p=layerDropouts[i])
      }
    }
  },
  forward = function(x) {
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](self[[str_c("layer_",i)]](x))
      if(self$Dropout){
        x <- self[[str_c("layerDropout_",i)]](x)
      }
    }
    x
  },
  fforward = function(x,state){
    for(i in seq_along(self$layerSizes)){
      x <- self$layerTransforms[[i]](
        nnf_linear(
          x, 
          weight = self[[str_c("layer_",i)]]$constructWeight(state[[str_c("layer_",i,".weightPar")]]), 
          bias = self[[str_c("layer_",i)]]$constructBias(state[[str_c("layer_",i,".biasPar")]])
          )
        )
    }
    x
  }
)





