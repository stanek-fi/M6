prepareBaseModel <- function(baseModel, x) {
  state <- baseModel$state_dict()
  
  baseModel$eval()
  output <- baseModel(x)
  outputControl <- baseModel$fforward(x,state)
  meanDiff <- as.array(mean(abs(output - outputControl)))
  if(meanDiff>1e-10){
    stop("Supplied fforward method of the baseModel does not match the forward method.")
  }
  baseModel$train()
  
  baseModel$outputSize <- output$size(2)
  
  baseModel$stateStructure <- rapply(state, function(x) dim(x), how = "list")
  
  baseModel$flattenState <- function(state) {
    torch_cat(rapply(state, function(x) x$view(-1), how = "unlist"))
  }
  
  baseModel$unflattenState <- function(stateFlat, stateStructure) {
    counter <- 1
    rapply(stateStructure , function(dimensions){
      totsize <- prod(dimensions)
      out <- stateFlat[(counter):(counter - 1 + totsize)]$view(dimensions)
      counter <<- counter + totsize
      return(out)
    }, how = "list")
  }
  
  stateControl <- baseModel$unflattenState(baseModel$flattenState(state), baseModel$stateStructure)
  stateControlEqual <- unlist(mapply(function(x,y) {
    all.equal(as.array(x), as.array(y))
  }, x=state, y=stateControl), recursive = T)
  if(!all(stateControlEqual == "TRUE")){
    stop("State transformation functions failed.")
  }
  
  baseModel$stateSize <- baseModel$flattenState(state)$size()
  
  return(baseModel)
}

MetaModel = nn_module(
  initialize = function(baseModel, xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0, initMesaRange = 0, initMetaRange = 1, mesaLayerWeightInit = NULL, metaLayerWeightInit = NULL, allowMetaStructure = NULL) {
    self$fforward <- baseModel$fforward
    self$stateStructure <- baseModel$stateStructure
    self$flattenState <- baseModel$flattenState
    self$unflattenState <- baseModel$unflattenState
    self$outputSize <- baseModel$outputSize
    self$stateSize <- baseModel$stateSize
    self$xtypeSize <- ncol(xtype)
    self$mesaParameterSize <- mesaParameterSize
    self$allowBias <- allowBias
    if(is.null(allowMetaStructure)){
      allowMetaStructure <- rapply(baseModel$stateStructure, function(x) torch_tensor(array(T, dim = x)), how = "list")
    }else{
      if(!identical(rapply(allowMetaStructure, function(x) dim(x), how = "list"), baseModel$stateStructure)){
        stop("invalid supplied allowMetaStructure")
      }
    }
    self$allowMetaVector <- as.array(baseModel$flattenState(allowMetaStructure)) # * for partial connection
    if(is.null(mesaLayerWeightInit)){
      self$mesaLayerWeight <- nn_parameter(torch_tensor(matrix(runif(self$mesaParameterSize * self$xtypeSize, -initMesaRange, initMesaRange), nrow = self$mesaParameterSize, ncol = self$xtypeSize), dtype = torch_float()))
    }else{
      if(!((nrow(mesaLayerWeightInit) == self$mesaParameterSize) & (ncol(mesaLayerWeightInit) == self$xtypeSize))){
        stop("ivalid supplied mesaLayerWeightInit")
      }
      self$mesaLayerWeight <- nn_parameter(torch_tensor(mesaLayerWeightInit, dtype = torch_float()))
      # self$mesaLayerWeight <- torch_tensor(mesaLayerWeightInit, dtype = torch_float())
    }
    if(is.null(metaLayerWeightInit)){
      # self$metaLayerWeight <- nn_parameter(torch_tensor(matrix(runif(baseModel$stateSize * mesaParameterSize, -initMetaRange, initMetaRange), nrow = baseModel$stateSize), dtype = torch_float()))
      self$metaLayerWeight <- nn_parameter(torch_tensor(matrix(runif(sum(self$allowMetaVector) * self$mesaParameterSize, -initMetaRange, initMetaRange), nrow = sum(self$allowMetaVector)), dtype = torch_float()))  # * for partial connection
    }else{
      # if(!((nrow(metaLayerWeightInit) == baseModel$stateSize) & (ncol(metaLayerWeightInit) == self$mesaParameterSize))){
      if (!((nrow(metaLayerWeightInit) == sum(self$allowMetaVector)) & (ncol(metaLayerWeightInit) == self$mesaParameterSize))) { # * for partial connection
        stop("ivalid supplied metaLayerWeightInit")
      }
      self$metaLayerWeight <- nn_parameter(torch_tensor(metaLayerWeightInit, dtype = torch_float()))
      # self$metaLayerWeight <- torch_tensor(metaLayerWeightInit, dtype = torch_float())
    }
    if(self$allowBias){
      self$metaLayerBias <- nn_parameter(torch_tensor(rep(0, self$stateSize), dtype = torch_float()))
    }
    self$baseState <- as.array(self$flattenState(baseModel$state_dict())) #storing it array to avoid odd ourside reference error. Update...it is caused by inbaility to conect references after saving the model...probably not worth fixing
    self$dropout <- nn_dropout(p = pDropout)
  },

  forward = function(x,xtype, ...) {
    if(is.null(horizon)){ #TODO: this is ugly as hell...you need to think about how to structure it better
      xout <- torch_zeros(nrow(x),self$outputSize)
    }else{
      xout <- torch_zeros(nrow(x),length(horizon))
    }
    iscoalesced = try({xtype$is_coalesced()},silent = T)
    baseState <- torch_tensor(self$baseState)

    columns <- xtype$indices()[2,]
    uniqueColumns <- unique(as.array(columns))
    for(i in uniqueColumns){
      indices <- columns == i
      xtypei <- torch_tensor(replace(numeric(xtype$size(2)), i + 1, 1)) #must add one to correct for zero indexing
      mesaState <- nnf_linear(xtypei, self$mesaLayerWeight)
      if(self$allowBias){
        # metaStateFlatDiff <- self$metaLayerBias + self$metaLayer(mesaState)
        # metaStateFlatDiff <- self$metaLayerBias + nnf_linear(mesaState, self$metaLayerWeight)
        temp <- torch_tensor(rep(0,self$stateSize))  # * for partial connection
        temp[self$allowMetaVector] <- nnf_linear(mesaState, self$metaLayerWeight)  # * for partial connection
        metaStateFlatDiff <- self$metaLayerBias + temp  # * for partial connection
      }else{
        # metaStateFlatDiff <- self$metaLayer(mesaState)
        # metaStateFlatDiff <- nnf_linear(mesaState, self$metaLayerWeight)
        temp <- torch_tensor(rep(0,self$stateSize))  # * for partial connection
        temp[self$allowMetaVector] <- nnf_linear(mesaState, self$metaLayerWeight)  # * for partial connection
        metaStateFlatDiff <- temp  # * for partial connection
      }
      metaStateFlat <-  baseState + self$dropout(metaStateFlatDiff)
      metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
      xout[indices] <- self$fforward(x[indices,],metaState, ...)
      # xout[indices,] <- self$fforward(x[indices,],metaState, horizon = horizon)
    }

    xout
  },

  MesaModel = function(metaModel) {

    metaState <- metaModel$state_dict()

    nn_module(
      initialize = function() {
        self$fforward <- metaModel$fforward
        self$stateStructure <- metaModel$stateStructure
        self$unflattenState <- metaModel$unflattenState
        self$flattenState <- metaModel$flattenState
        self$mesaParameterSize <- metaModel$mesaParameterSize
        self$allowBias <- metaModel$allowBias
        if(self$allowBias){
          self$metaLayerBias <- torch_tensor(as.array(metaState$metaLayerBias),dtype = torch_float()) #nasty trick to make it leaf
        }
        self$metaLayerWeight <- torch_tensor(as.array(metaState$metaLayerWeight),dtype = torch_float()) #nasty trick to make it leaf
        self$baseState <- metaModel$baseState
        self$mesaState <- nn_parameter(torch_tensor(rep(0, self$mesaParameterSize), dtype = torch_float()))
        self$stateSize <- metaModel$stateSize
        self$allowMetaVector <- metaModel$allowMetaVector
      },
      forward = function(x, ...) {
        mesaState <- self$mesaState
        baseState <- torch_tensor(self$baseState)
        if(self$allowBias){
          # metaStateFlatDiff <- self$metaLayerBias + nnf_linear(mesaState,self$metaLayerWeight)
          temp <- torch_tensor(rep(0,self$stateSize))  # * for partial connection
          temp[self$allowMetaVector] <- nnf_linear(mesaState, self$metaLayerWeight)  # * for partial connection
          metaStateFlatDiff <- self$metaLayerBias + temp  # * for partial connection
        }else{
          # metaStateFlatDiff <- nnf_linear(mesaState,self$metaLayerWeight)
          temp <- torch_tensor(rep(0,self$stateSize))  # * for partial connection
          temp[self$allowMetaVector] <- nnf_linear(mesaState, self$metaLayerWeight)  # * for partial connection
          metaStateFlatDiff <- temp  # * for partial connection
        }
        metaStateFlat <-  baseState + metaStateFlatDiff
        metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
        self$fforward(x,metaState, ...)
      }
    )
  }
)


# MetaModel = nn_module(
#   initialize = function(baseModel, xtype, mesaParameterSize = 1, allowBias = T, pDropout = 0, initMesaRange = 0, initMetaRange = 1) {
#     self$fforward <- baseModel$fforward
#     self$stateStructure <- baseModel$stateStructure
#     self$flattenState <- baseModel$flattenState
#     self$unflattenState <- baseModel$unflattenState
#     self$outputSize <- baseModel$outputSize 
#     self$stateSize <- baseModel$stateSize
#     self$xtypeSize <- ncol(xtype)
#     self$mesaParameterSize <- mesaParameterSize
#     self$allowBias <- allowBias
#     self$mesaLayerWeight <- nn_parameter(torch_tensor(matrix(runif(self$mesaParameterSize * self$xtypeSize, -initMesaRange, initMesaRange), nrow = self$mesaParameterSize, ncol = self$xtypeSize), dtype = torch_float()))
#     # self$metaLayer <- nn_linear(mesaParameterSize, baseModel$stateSize, bias = F)
#     self$metaLayerWeight <- nn_parameter(torch_tensor(matrix(runif(baseModel$stateSize * mesaParameterSize, -initMetaRange, initMetaRange), nrow = baseModel$stateSize), dtype = torch_float()))
#     if(self$allowBias){
#       self$metaLayerBias <- nn_parameter(torch_tensor(rep(0, self$stateSize), dtype = torch_float()))
#     }
#     self$baseState <- as.array(self$flattenState(baseModel$state_dict())) #storing it array to avoid odd ourside reference error. Update...it is caused by inbaility to conect references after saving the model...probably not worth fixing
#     self$dropout <- nn_dropout(p = pDropout)
#   },
  
#   forward = function(x,xtype) {
#     xout <- torch_zeros(nrow(x),self$outputSize)
#     iscoalesced = try({xtype$is_coalesced()},silent = T)
#     baseState <- torch_tensor(self$baseState)
    
#     if(iscoalesced == TRUE){
#       columns <- xtype$indices()[2,]
#       uniqueColumns <- unique(as.array(columns))
#       for(i in uniqueColumns){
#         indices <- columns == i
#         xtypei <- torch_tensor(replace(numeric(xtype$size(2)), i + 1, 1)) #must add one to correct for zero indexing
#         mesaState <- nnf_linear(xtypei, self$mesaLayerWeight) 
#         if(self$allowBias){
#           # metaStateFlatDiff <- self$metaLayerBias + self$metaLayer(mesaState) 
#           metaStateFlatDiff <- self$metaLayerBias + nnf_linear(mesaState, self$metaLayerWeight)
#         }else{
#           # metaStateFlatDiff <- self$metaLayer(mesaState)
#           metaStateFlatDiff <- nnf_linear(mesaState, self$metaLayerWeight)
#         }
#         metaStateFlat <-  baseState + self$dropout(metaStateFlatDiff)
#         metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
#         xout[indices] <- self$fforward(x[indices,],metaState)
#       }
#     }else{
#       # for(i in seq_len(ncol(xtype))){
#       #   indices <- xtype[,i]>0
#       #   if(as.numeric(indices$max())>0){
#       #     mesaState <- nnf_linear(xtype[indices,][1,], self$mesaLayerWeight)
#       #     metaStateFlat <-  self$metaLayerBias + self$metaLayer(mesaState)
#       #     metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
#       #     xout[indices] <- self$fforward(x[indices,],metaState)
#       #   }
#       # }
#       stop("Currently works only with sparse xtype.")
#     }

#     xout
#   },
  
#   MesaModel = function(metaModel) { 
    
#     metaState <- metaModel$state_dict()
    
#     nn_module(
#       initialize = function() {
#         self$fforward <- metaModel$fforward
#         self$stateStructure <- metaModel$stateStructure
#         self$unflattenState <- metaModel$unflattenState
#         self$flattenState <- metaModel$flattenState
#         self$mesaParameterSize <- metaModel$mesaParameterSize
#         self$allowBias <- metaModel$allowBias
#         if(self$allowBias){
#           self$metaLayerBias <- torch_tensor(as.array(metaState$metaLayerBias),dtype = torch_float()) #nasty trick to make it leaf
#         }
#         self$metaLayerWeight <- torch_tensor(as.array(metaState$metaLayer.weight),dtype = torch_float()) #nasty trick to make it leaf
#         self$baseState <- metaModel$baseState
#         self$mesaState <- nn_parameter(torch_tensor(rep(0, self$mesaParameterSize), dtype = torch_float()))
#       },
#       forward = function(x) {
#         mesaState <- self$mesaState
#         baseState <- torch_tensor(self$baseState)
#         if(self$allowBias){
#           metaStateFlatDiff <- self$metaLayerBias + nnf_linear(mesaState,self$metaLayerWeight)
#         }else{
#           metaStateFlatDiff <- nnf_linear(mesaState,self$metaLayerWeight)
#         }
#         metaStateFlat <-  baseState + metaStateFlatDiff
#         metaState <- self$unflattenState(metaStateFlat, self$stateStructure)
#         self$fforward(x,metaState)
#       }
#     )
#   }
# )

