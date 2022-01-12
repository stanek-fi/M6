library(torch)

N <- 1000
NumGroups <- 2
# thetas <- rep(1,NumGroups)
thetas <- c(-1,1)
train_split = 0.8

type = as.factor(sample(1:NumGroups,size=N ,replace = T))
xtype <- model.matrix(~type-1)
thetas <- xtype %*% thetas
x <- as.matrix(rnorm(N),N,1)
y <- x * thetas

sample_indices = sample(1:N, size=N * train_split)
x_train = as.matrix(x[sample_indices,])
y_train = as.matrix(y[sample_indices,])
x_test = as.matrix(x[-sample_indices,])
y_test = as.matrix(y[-sample_indices,])
xtype_train = as.matrix(xtype[sample_indices,])
xtype_test = as.matrix(xtype[-sample_indices,])


x_train = torch_tensor(x_train, dtype = torch_float())
y_train = torch_tensor(y_train, dtype = torch_float())
x_test = torch_tensor(x_test, dtype = torch_float())
y_test = torch_tensor(y_test, dtype = torch_float())
xtype_train = torch_tensor(xtype_train, dtype = torch_float())
xtype_test = torch_tensor(xtype_test, dtype = torch_float())



# net = nn_module(
#   initialize = function() {
#     self$lin1 <- nn_linear(1, 1, bias = F)
#   },
#   forward = function(x,xtype) {
#     x <- self$lin1(x)
#     x
#   }
# )

xtype <- xtype_train
x <- x_train
net = nn_module(
  initialize = function() {
    # self$lin1 <- nn_linear(1, 1, bias = F)
    self$linIn <- nn_linear(2, 1)
  },
  forward = function(x,xtype) {

    xout <- torch_zeros(nrow(x),1)
    for(i in seq_len(ncol(xtype))){
      indices <- xtype[,i]>0
      if(as.numeric(indices$max())>0){
        theta <- self$linIn(xtype[indices,][1,])
        theta <- torch_reshape(theta,c(1,1))
        xout[indices] <- nnf_linear(x[indices,],theta)
      }
    }
    xout
  }
  # forward = function(x,xtype) {
  #   thetas <- self$linIn(xtype)
  #   xout <- torch_zeros(nrow(x),1)
  #   for(i in 1:nrow(thetas)){
  #     theta <- thetas[i,]
  #     theta <- torch_reshape(theta,c(1,1))
  #     # print(theta)
  #     xout[i,] <- nnf_linear(x[i,],theta)
  #   }
  #   xout
  # }
  # forward = function(x,xtype) {
  #   thetas <- self$linIn(xtype)
  #   theta <- thetas[1,]
  #   theta <- torch_reshape(theta,c(1,1))
  #   # print(theta)
  #   x <- nnf_linear(x,theta)
  #   x
  # }
)


model <- net()
criterion = nn_mse_loss()
optimizer = optim_adam(model$parameters, lr = 0.01)
epochs = 300
# Train the net
for(i in 1:epochs){
  
  optimizer$zero_grad()
  y_pred = model(x_train,xtype_train)
  loss = criterion(y_pred, y_train)
  loss$backward()
  optimizer$step()
  
  # Check Training
  # if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  # }
}

model$parameters








