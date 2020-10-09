rm(list = ls())

setwd("C://Users//User//Desktop//과제//3학년 2학기//딥러닝//HW1")
getwd()
stock_data = read.csv('stock.csv', sep = ",", header = TRUE)
attach(stock_data)
head(stock_data)

### Construct the dummy variable matrix for Month variable
dummy_Month <-  as.factor(Month)
dummy_Month
dummy_matrix <-  model.matrix(~ dummy_Month)
dummy_matrix <-  dummy_matrix[,-c(1)]
dummy_matrix
dim(dummy_matrix)

### Construct the design matrix X
X <- model.matrix(~ Interest + Unemployment + dummy_Month)
X
dim(X)

### Calculate beta hat / standard error of beta hat
y <- Stock
beta.hat <- solve(t(X)%*%X)%*%t(X)%*%y
beta.hat

sigmasq.hat <- as.numeric(t(y-X%*%beta.hat)%*%(y-X%*%beta.hat)/(24-14))
se <- sqrt(diag(solve(t(X)%*%X))*sigmasq.hat)
se

### Use lm function
model <- lm(Stock ~ Interest + Unemployment + as.factor(Month))

model$coefficients
beta.hat

summary(model)
se
