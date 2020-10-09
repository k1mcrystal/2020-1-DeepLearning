rm(list = ls())

###############################
## (a) ##
###############################

### Initialize parameters
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

### Input and target values
input1 = 0.05
input2 = 0.10
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)

### Set learning rate
gamma = 0.5

### define functions
sigmoid = function(z){return (1 / (1 + exp(-z)))}

forwardProp = function(input, w, b){
  neth1 = w[1]*input[1] + w[2]*input[2] + b[1]
  neth2 = w[3]*input[1] + w[4]*input[2] + b[1]
  outh1 = sigmoid(neth1)
  outh2 = sigmoid(neth2)
  
  neto1 = w[5]*outh1 + w[6]*outh2 + b[2]
  neto2 = w[7]*outh1 + w[8]*outh2 + b[2]
  outo1 = sigmoid(neto1)
  outo2 = sigmoid(neto2)
  
  res = c(outh1, outh2, outo1, outo2)
  return(res)
}


error = function(res,out){
  err = 0.5*(out[1] - res[3])^2 + 0.5*(out[2] - res[4])^2
  return(err)
}

### Forward Propagation
res = forwardProp(input, w, b)
outh1 = res[1] ; outh2 = res[2] ; outo1 = res[3] ; outo2 = res[4]
err = error(res, out)

### Backward Propagation
backwardProp = function(res, input, out, gamma){
  ## w5 to w8
  # compute dE_dw5
  dE_douto1 = -(out[1]-outo1)
  douto1_dneto1 = outo1*(1-outo1)
  dneto1_dw5 = outh1
  dE_dw5 = dE_douto1 * douto1_dneto1 * dneto1_dw5
  # compute dE_dw6
  dneto1_dw6 = outh2
  dE_dw6 = dE_douto1 * douto1_dneto1 * dneto1_dw6
  # compute dE_dw7
  dE_douto2 = -(out[2]-outo2)
  douto2_dneto2 = outo2*(1-outo2)
  dneto2_dw7 = outh1
  dE_dw7 = dE_douto2 * douto2_dneto2 * dneto2_dw7
  # compute dE_dw8
  dneto2_dw8 = outh2
  dE_dw8 = dE_douto2 * douto2_dneto2 * dneto2_dw8
  
  ## w1 to w4
  # compute dE_dw1
  dneto1_douth1 = w5
  dneto2_douth1 = w7
  dE_douth1 = dE_douto1 * douto1_dneto1 * dneto1_douth1 + dE_douto2 * douto2_dneto2 * dneto2_douth1
  douth1_dneth1 = outh1 * (1-outh1)
  dneth1_dw1 = input[1]
  dE_dw1 = dE_douth1 * douth1_dneth1 * dneth1_dw1
  # compute dE_dw2
  dneth1_dw2 = input[2]
  dE_dw2 = dE_douth1 * douth1_dneth1 * dneth1_dw2
  # compute dE_dw3
  dneto1_douth2 = w6
  dneto2_douth2 = w8
  dE_douth2 = dE_douto1 * douto1_dneto1 * dneto1_douth2 + dE_douto2 * douto2_dneto2 * dneto2_douth2
  douth2_dneth2 = outh2 * (1-outh2)
  dneth2_dw3 = input[1]
  dE_dw3 = dE_douth2 * douth2_dneth2 * dneth2_dw3
  # compute dE_dw4
  dneth2_dw4 = input[2]
  dE_dw4 = dE_douth2 * douth2_dneth2 * dneth2_dw4
  
  # compute dE_db2
  dE_db2 = dE_douto1*douto1_dneto1*1 + dE_douto2*douto2_dneto2*1
  
  # compute dE_db1
  dE_db1 = dE_douto1*douto1_dneto1*dneto1_douth1*douth1_dneth1*1 + dE_douto2*douto2_dneto2*dneto2_douth2*douth2_dneth2*1
  
  ## Update all parameters via a gradient decent
  w1 = w1 - gamma*dE_dw1
  w2 = w2 - gamma*dE_dw2
  w3 = w3 - gamma*dE_dw3
  w4 = w4 - gamma*dE_dw4
  w5 = w5 - gamma*dE_dw5
  w6 = w6 - gamma*dE_dw6
  w7 = w7 - gamma*dE_dw7
  w8 = w8 - gamma*dE_dw8
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b1 = b1 - gamma*dE_db1
  b2 = b2 - gamma*dE_db2 
  return(c(w, b))
}
backwardProp(res, input, out, gamma)[1:8]


###############################
## (b) ##
###############################
numIter = 1

### Implement Forward-backward propagation
err = c()
for(i in 1:numIter){
  
  ### forward
  res = forwardProp(input, w, b)
  outh1 = res[1]; outh2 = res[2]; outo1 = res[3]; outo2 = res[4]
  
  ### compute error
  err[i] = error(res, out)
  
  ### backward propagation
  ## update w_5, w_6, w_7, w_8, b2 
  # compute dE_dw5
  dE_douto1 = -( out[1] - outo1 )
  douto1_dneto1 = outo1*(1-outo1)
  dneto1_dw5 = outh1
  dE_dw5 = dE_douto1*douto1_dneto1*dneto1_dw5
  
  # compute dE_dw6
  dneto1_dw6 = outh2
  dE_dw6 = dE_douto1*douto1_dneto1*dneto1_dw6
  
  # compute dE_dw7
  dE_douto2 = -( out[2] - outo2 )
  douto2_dneto2 = outo2*(1-outo2)
  dneto2_dw7 = outh1
  dE_dw7 = dE_douto2*douto2_dneto2*dneto2_dw7
  
  # compute dE_dw8
  dneto2_dw8 = outh2
  dE_dw8 = dE_douto2*douto2_dneto2*dneto2_dw8
  
  # compute dE_db2
  dE_db2 = dE_douto1*douto1_dneto1*1 + dE_douto2*douto2_dneto2*1
  
  ## update w_1, w_2, w_3, w_4, b1 
  # compute dE_douth1 first
  dneto1_douth1 = w5
  dneto2_douth1 = w7
  dE_douth1 = dE_douto1*douto1_dneto1*dneto1_douth1 + dE_douto2*douto2_dneto2*dneto2_douth1
  
  # compute dE_douth2 first
  dneto1_douth2 = w6
  dneto2_douth2 = w8
  dE_douth2 = dE_douto1*douto1_dneto1*dneto1_douth2 + dE_douto2*douto2_dneto2*dneto2_douth2 
  
  # compute dE_dw1    
  douth1_dneth1 = outh1*(1-outh1)
  dneth1_dw1 = input[1]
  dE_dw1 = dE_douth1*douth1_dneth1*dneth1_dw1
  
  # compute dE_dw2
  dneth1_dw2 = input[2]
  dE_dw2 = dE_douth1*douth1_dneth1*dneth1_dw2
  
  # compute dE_dw3
  douth2_dneth2 = outh2*(1-outh2)
  dneth2_dw3 = input[1] 
  dE_dw3 = dE_douth2*douth2_dneth2*dneth2_dw3
  
  # compute dE_dw4
  dneth2_dw4 = input[2]
  dE_dw4 = dE_douth2*douth2_dneth2*dneth2_dw4  
  
  # compute dE_db1
  dE_db1 = dE_douto1*douto1_dneto1*dneto1_douth1*douth1_dneth1*1 + dE_douto2*douto2_dneto2*dneto2_douth2*douth2_dneth2*1
  
  ### update all parameters via a gradient descent 
  w1_b = w1 - gamma*dE_dw1
  w2_b = w2 - gamma*dE_dw2
  w3_b = w3 - gamma*dE_dw3
  w4_b = w4 - gamma*dE_dw4
  w5_b = w5 - gamma*dE_dw5
  w6_b = w6 - gamma*dE_dw6
  w7_b = w7 - gamma*dE_dw7
  w8_b = w8 - gamma*dE_dw8
  b1_b = b1 - gamma*dE_db1
  b2_b = b2 - gamma*dE_db2    
  
  w_b = c(w1_b, w2_b, w3_b, w4_b, w5_b, w6_b, w7_b, w8_b)
  b_b= c(b1_b, b2_b)
  
  
}
w_b
backwardProp(res, input, out, gamma)[1:8]

###############################
## (c) ##
###############################

### 10000 iterations
## learning rate = 0.1
# Initialize parameters
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

input1 = 0.05
input2 = 0.10
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)

gamma = 0.1

res = forwardProp(input, w, b)
err = c()
for (i in 1:10000) {
  param = backwardProp(res, input, out, gamma)
  w = param[1:8]
  b = param[9:10]
  res = forwardProp(input, w, b)
  err[i] = error(res, out)
  w1 = w[1] ; w2 = w[2] ; w3 = w[3] ; w4 = w[4] ; w5 = w[5] ; w6 = w[6] ; w7 = w[7] ; w8 = w[8]
  b1 = b[1] ; b2 = b[2]
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
}
err0.1 <- err
res0.1 <- res

## learning rate = 0.6
# Initialize parameters
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

# Input and target values
input1 = 0.05
input2 = 0.10
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)

gamma = 0.6

res = forwardProp(input, w, b)
err = c()
for (i in 1:10000) {
  param = backwardProp(res, input, out, gamma)
  w = param[1:8]
  b = param[9:10]
  res = forwardProp(input, w, b)
  err[i] = error(res, out)
  w1 = w[1] ; w2 = w[2] ; w3 = w[3] ; w4 = w[4] ; w5 = w[5] ; w6 = w[6] ; w7 = w[7] ; w8 = w[8]
  b1 = b[1] ; b2 = b[2]
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
}
err0.6 <- err
res0.6 <- res

## learning rate = 1.2
# Initialize parameters
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55
w = c(w1, w2, w3, w4, w5, w6, w7, w8)

b1 = 0.35
b2 = 0.60
b = c(b1, b2)

input1 = 0.05
input2 = 0.10
input = c(input1, input2)

out1 = 0.01
out2 = 0.99
out = c(out1, out2)

gamma = 1.2

res = forwardProp(input, w, b)
err = c()
for (i in 1:10000) {
  param = backwardProp(res, input, out, gamma)
  w = param[1:8]
  b = param[9:10]
  res = forwardProp(input, w, b)
  err[i] = error(res, out)
  w1 = w[1] ; w2 = w[2] ; w3 = w[3] ; w4 = w[4] ; w5 = w[5] ; w6 = w[6] ; w7 = w[7] ; w8 = w[8]
  b1 = b[1] ; b2 = b[2]
  w = c(w1, w2, w3, w4, w5, w6, w7, w8)
  b = c(b1, b2)
  
}
err1.2 <- err
res1.2 <- res

plot(err0.1, type = 'l', col = 'red', ylab = 'error', xlab = 'iteration', ylim = c(0,0.3),  xlim = c(0,10000), main = 'error rate')
par(new = T)
plot(err0.6, type = 'l', col = 'black', ylab = 'error', xlab = 'iteration', ylim = c(0,0.3),  xlim = c(0,10000))
par(new = T)
plot(err1.2, type = 'l', col = 'blue', ylab = 'error', xlab = 'iteration', ylim = c(0,0.3),  xlim = c(0,10000))
legend(x = 7500, y = 0.28, c("0.1", "0.6", "1.2"), cex = 0.9, lty = c(1,1,1), col = c("red", "black", "blue"))
iteration.df = data.frame(gamma = c(0.1, 0.6, 1.2), o1 = rep(0,3), o2 = rep(0,3)) 
iteration.df[1,2:3] = res0.1[3:4]
iteration.df[2,2:3] = res0.6[3:4]
iteration.df[3,2:3] = res1.2[3:4]
iteration.df
out
res0.1[3:4]
res0.6[3:4]
res1.2[3:4]
