rm(list = ls())

# import
library("kohonen")
data("wines")
head(wines)

# 2*2 
mygrid = somgrid(2,2,"hexagonal")
som.wines = som(scale(wines), grid = mygrid)
som.wines
dim(getCodes(som.wines))
plot(som.wines)

par(mfrow = c(1,1))
plot(som.wines, type = "changes", main = "2*2")

# 4*4
mygrid = somgrid(4,4,"hexagonal")
som.wines = som(scale(wines), grid = mygrid)
som.wines
dim(getCodes(som.wines))
plot(som.wines)

par(mfrow = c(1,1))
plot(som.wines, type = "changes", main = "4*4")

# 6*6
mygrid = somgrid(6,6,"hexagonal")
som.wines = som(scale(wines), grid = mygrid)
som.wines
dim(getCodes(som.wines))
plot(som.wines)

par(mfrow = c(1,1))
plot(som.wines, type = "changes", main = "6*6")

# 10*10
mygrid = somgrid(10,10,"hexagonal")
som.wines = som(scale(wines), grid = mygrid)
som.wines
dim(getCodes(som.wines))
plot(som.wines)

par(mfrow = c(1,1))
plot(som.wines, type = "changes", main = "10*10")

# Use vintages data set
training = sample(nrow(wines), 150)
Xtraining = scale(wines[training,])
Xtest = scale(wines[-training,],center = attr(Xtraining, "scaled:center"),scale = attr(Xtraining, "scaled:scale"))
trainingdata = list(measurements = Xtraining, vintages = vintages[training])
testdata = list(measurements = Xtest, vintages = vintages[-training])

pred <- c()
grid <- c(2,3,4,5,6,7,8,9,10)
for (i in grid){
  mygrid = somgrid(i,i,"hexagonal") 
  som.wines = supersom(trainingdata, grid = mygrid)
  som.prediction = predict(som.wines, newdata = testdata)
  res <- sum(diag(table(vintages[-training], som.prediction$predictions[["vintages"]])))/length(testdata$vintages)
  pred <- c(pred,res)}
data.frame(grid,pred)

