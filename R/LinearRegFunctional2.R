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


x <- x_train
net = nn_module(
  initialize = function() {
    self$lin1 <- nn_linear(1, 3)
    self$lin2 <- nn_linear(3, 3)
    self$lin3 <- nn_linear(3, 1)
  },
  forward = function(x) {
    x <- self$lin1(x)
    x <- nnf_relu(x)
    x <- self$lin2(x)
    x <- nnf_relu(x)
    x <- self$lin3(x)
    x
  },
  fforward = function(x,state) {
    x <- nnf_linear(x,state$lin1.weight,state$lin1.bias)
    x <- nnf_relu(x)
    x <- nnf_linear(x,state$lin2.weight,state$lin2.bias)
    x <- nnf_relu(x)
    x <- nnf_linear(x,state$lin3.weight,state$lin3.bias)
    x
  }
)

model <- net()
state <- model$state_dict()
model(x_train) - model$fforward(x,state)


state



par
parlong <- torch_cat(rapply(par, function(x) x$view(-1), how = "unlist"))


parStructure <- rapply(par, function(x) dim(x), how = "list")


start <- 1
par2 <- rapply(parStructure , function(dimensions){
  totsize <- prod(dimensions)
  out <- parlong[(start):(start - 1 + totsize)]$view(dimensions)
  start <<- start + totsize
  return(out)
}, how = "list")

model$load_state_dict(par2)






torch_stack(lapply(par, function(x) x$view(-1)))