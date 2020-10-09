rm(list = ls())

# (a)

## Simulate correlated data sets
N=150
P=50
X = matrix(NA, nrow = N, ncol = P)

covmat = matrix(rnorm(P^2, sd = 2), nrow = P)
covmat = covmat+t(covmat)
U = eigen(covmat)$vectors
D = diag(rexp(P, rate = 10))
covmat = U %*% D %*% t(U)

library(mvtnorm)
set.seed(3)
for(i in 1:N){X[i,]=rmvnorm(1,mean=rep(0,P), sigma = covmat)}
X = data.frame(X)
dim(X)
head(X, n = 2)

# true betas
betas.true=c(1,2,3,4,5,-1,-2,-3,-4,-5,rep(0,P-10))

# simulating "y"
sigma = 15.7  # arbitrary value
X = as.matrix(X)
y = X %*% betas.true + rnorm(N, mean=0, sd = sigma)

## Divide the data sets into training and test 
alldata = data.frame(cbind(y,X))
names(alldata)[1] <- "y"
head(alldata, n = 2)
train = alldata[1:100,]
test = alldata[101:150,]

## Fit the ordinary linear regression using training data sets
fit = lm(y~., data = train)
summary(fit)
betas.lm = coef(fit)

## Calulate VIFs
library(car)
vif(fit)


# (b)

## Find the best lambda

# Fit ridge (trying 100 different lambda values)
library(glmnet)
rr = glmnet(x = as.matrix(train[,-1]), y = as.numeric(train[,1]), alpha = 0, nlambda = 100)
plot(rr, xvar = "lambda", main = "Ridge Regression Betas for Different Values of the Tuning Parameter")

# Use 10-fold crossvalidation to find the best lambda
cv.rr = cv.glmnet(x = as.matrix(train[,-1]), y = as.numeric(train[,1]), alpha = 0, nfolds = 10, nlambda = 100)

# get cvmspe from best value of lambda
cvmspe.rr = min(cv.rr$cvm)
cvmspe.rr

# Get lambda and best rr fit
lambda.rr = cv.rr$lambda.min
lambda.rr

##  Compare the estimated Ridge regression coefficients and standard regression coefficients
betas.rr = coef(cv.rr, s = "lambda.min")
cbind(betas.rr, betas.lm)


# (c)

## Repeat (b) for lasso regression

# Fit lasso (trying 100 different lambda values)
lasso = glmnet(x = as.matrix(train[,-1]), y = as.numeric(train[,1]), alpha = 1, nlambda = 100)
plot(lasso, xvar = "lambda", main = "Lasso Regression Betas for Different Values of the Tuning Parameter")

# Use 10-fold crossvalidation to find the best lambda
cv.lasso = cv.glmnet(x = as.matrix(train[,-1]), y = as.numeric(train[,1]), alpha = 1, nfolds = 10)

# get cvmspe from best value of lambda
cvmspe.lasso = min(cv.lasso$cvm)
cvmspe.lasso

# Get lambda and best lasso fit
lambda.lasso = cv.lasso$lambda.min
lambda.lasso

## Compare the estimated Lasso regression coefficients and standard regression coefficients.
betas.lasso = coef(cv.lasso, s = "lambda.min")
cbind(betas.lasso, betas.lm)


# (d)

## Compare the mean square prediction errors (mspe) for linear model, Lasso, and Ridge.

# Linear model
yhat.lm = predict(fit, newdata = test)
mspe.lm = mean((test$y - yhat.lm)^2)
mspe.lm

# Ridge
yhat.rr = predict(cv.rr, s = "lambda.min", newx = as.matrix(test[,-1]))
mspe.rr = mean((test$y-yhat.rr)^2)
mspe.rr

# Lasso
yhat.lasso = predict(cv.lasso, newx = as.matrix(test[,-1]), s = "lambda.min")
mspe.lasso = mean((test$y-yhat.lasso)^2)
mspe.lasso


# (e)

##  Compare the Lasso coefficients obtained from the uncorrelated case and from the correlated case.

# Correlated case
Correlated = betas.lasso

# Uncorrelated case

# Simulate correlated data sets
N=150
P=50
X = matrix(NA, nrow = N, ncol = P)

covmat = diag(P)
set.seed(3)

for(i in 1:N){X[i,]=rmvnorm(1,mean=rep(0,P), sigma = covmat)}
X = data.frame(X)
dim(X)
head(X, n = 2)

# true betas
betas.true=c(1,2,3,4,5,-1,-2,-3,-4,-5,rep(0,P-10))

# simulating "y"
sigma = 15.7  # arbitrary value
X = as.matrix(X)
y = X %*% betas.true + rnorm(N, mean=0, sd = sigma)

# Divide the data sets into training and test 
alldata = data.frame(cbind(y,X))
names(alldata)[1] <- "y"
head(alldata, n = 2)
train = alldata[1:100,]
test = alldata[101:150,]

# Fit lasso (trying 100 different lambda values)
lasso = glmnet(x = as.matrix(train[,-1]), y = as.numeric(train[,1]), alpha = 1, nlambda = 100)

# Use 10-fold crossvalidation to find the best lambda
cv.lasso = cv.glmnet(x = as.matrix(train[,-1]), y = as.numeric(train[,1]), alpha = 1, nfolds = 10, nlambda = 100)

# get cvmspe from best value of lambda
cvmspe.lasso = min(cv.lasso$cvm)
cvmspe.lasso

# Get lambda and best rr fit
lambda.lasso = cv.lasso$lambda.min
lambda.lasso

# Compare
Uncorrelated = coef(cv.lasso, s = "lambda.min")
cbind(Correlated, Uncorrelated)


# (f)

rm(list = ls())
# Simulate logistic regression
N=150
P=50

X = matrix(rnorm(N*P), nrow = N, ncol = P)

# True betas
betas.true=c(rep(2,10),rep(0,P-10))

## Simulating "y"
X = as.matrix(X)
eta.true = X%*%betas.true
mu.true = exp(eta.true)
mu.true
y = rpois(N, lambda = mu.true)
y

## Lasso
lasso = glmnet(x = X, y = y, family = "poisson", alpha = 1, nlambda = 100)

## Use 10-fold crossvalidation to find the best lambda
cv.lasso = cv.glmnet(x=X, y = y, alpha = 1, nfolds = 10)

## Get lambda and best lasso fit
lambda.lasso = cv.lasso$lambda.min
lambda.lasso

## Some plots
par(mfrow = c(1,2))
plot(cv.lasso)
abline(v = log(lambda.lasso))
plot(lasso, xvar = "lambda")
abline(v = log(lambda.lasso))

## Best estimates for best lambda
betas.lasso = coef(cv.lasso)
betas.lasso

