rm(list = ls())

###############################
## (a) ##
###############################

### Import data
data(iris)
iris_sub = iris[1:100, c(1,3,5)]
names(iris_sub) = c("sepal", "petal", "species")
head(iris_sub)

y = iris_sub[, 3]
x = iris_sub[, 1:2]
y = c(rep(-1, 50), rep(1, 50))
y

### Define functions
euclidean.norm = function (x) {sqrt(sum(x*x))}
distance.from.plane = function (z,w,b) {sum(z*w)+b}
classify.linear = function (x,w,b) {
  distances = apply(x, 1, distance.from.plane, w, b)
  return(ifelse(distances < 0, -1, +1))
}

### Define perceptron function

perceptron = function (x, y, learning.rate) {
  w = c(0,0)
  b = 0
  k = 0
  R = max(apply(x, 1, euclidean.norm))
  mark.complete = TRUE
  
  while (mark.complete) {
    mark.complete = FALSE
    yc = classify.linear(x,w,b)
    for (i in 1:nrow(x)) {
      if (y[i] != yc[i]) {
        w = w + learning.rate * y[i] * x[i,]
        b = b + learning.rate * y[i] * R^2
        k = k+1
        mark.complete = TRUE
      }
    }
  }
  return(list(w = w, b = b, k = k))
}

###############################
## (b) ##
###############################

p = perceptron(x, y, 1)
w = c(as.numeric(p$w)[1], as.numeric(p$w)[2])
b = p$b
k = p$k
w
b
k

###############################
## (c) ##
###############################
library(ggplot2)
ggplot(iris_sub, aes(x = sepal, y = petal))+
       geom_point(aes(colour = species, shape = species), size = 3)+
       xlab("Sepal length")+
       ylab("Petal length")+
       geom_abline(intercept = -b/w[2], slope = -w[1]/w[2], color = 'green', size = 1.2)
