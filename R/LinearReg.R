library(torch)

N <- 100
NumGroups <- 2
thetas <- rep(1,NumGroups)
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

net = nn_module(
  initialize = function() {
    self$lin1 <- nn_linear(1, 1, bias = F)
    self$linIn <- nn_linear(2, 1)
  },
  forward = function(x,xtype) {
    # theta <- self$linIn(xtype)
    # theta <- theta[1,]
    # theta <- torch_reshape(theta,c(1,1))
    # print(theta)
    
    state_dict <- self$lin1$state_dict()
    # state_dict[[1]] <- theta
    state_dict[[1]] <- state_dict[[1]] * -1
    self$lin1$load_state_dict(state_dict)
    
    x <- self$lin1(x)
    x
  }
)

net = nn_module(
  initialize = function() {
    self$linIn <- nn_linear(2, 1)
  },
  forward = function(x,xtype) {
    theta <- self$linIn(xtype)
    theta <- theta[1,]
    theta <- torch_reshape(theta,c(1,1))
    # print(theta)
    
    lin1 <- nn_linear(1, 1, bias = F)
    state_dict <- lin1$state_dict()
    state_dict[[1]] <- theta
    # state_dict[[1]] <- state_dict[[1]] * -1
    lin1$load_state_dict(state_dict)
    
    x <- lin1(x)
    x
  }
)




optimizer$zero_grad()
linIn$zero_grad()
lin1$zero_grad()

# linIn <- nn_linear(2, 1)
# theta <- linIn(xtype_train)
# theta <- theta[1,]
# theta <- torch_reshape(theta,c(1,1))

lin1 <- nn_linear(1, 1, bias = F)
lin1$parameters$weight
lin1$parameters$weight$grad
state_dict <- lin1$state_dict()
# state_dict[[1]] <- theta
par <- torch_tensor(matrix(1),dtype = torch_float(), requires_grad = T)
theta <- torch_tensor(matrix(1.5),dtype = torch_float(), requires_grad = T) + 0.5 * par
# theta <- torch_tensor(matrix(2),dtype = torch_float(), requires_grad = T)
# state_dict[[1]] <- torch_tensor(matrix(2),dtype = torch_float())
# state_dict[[1]] <- theta
# lin1$load_state_dict(state_dict)
# lin1$weight=theta
# lin1$weight$is_leaf=T
# lin1$parameters$weight$is_leaf=T
# lin1$weight=torch_tensor(matrix(1),dtype = torch_float(), requires_grad = T)
# lin1$parameters$weight=NULL

# y_pred = x_train * theta
y_pred =lin1(x_train)
y_pred =nnf_linear(x_train, weight = theta)

loss = criterion(y_pred, y_train)
loss$backward(create_graph = T)

lin1$parameters$weight
lin1$parameters$weight$grad
theta$grad
par$grad



linIn$parameters$weight$grad




lin1 <- nn_linear(1, 1, bias = F)
lin1$parameters$weight
lin1$parameters$weight$grad
y_pred =lin1(x_train)
loss = criterion(y_pred, y_train)
loss$backward()
lin1$parameters$weight$grad

lin1$zero_grad()



# lin1 <- nn_linear(1, 1, bias = F)
# lin1(x_train)


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
  if(i %% 10 == 0){
    cat(" Epoch:", i,"Loss: ", loss$item(),"\n")
  }
}

model$parameters





