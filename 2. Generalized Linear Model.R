admissions <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
head(admissions)

### fit glm function
fit = glm(admit ~ gpa, family = binomial, data = admissions)
summary(fit)

### get estimated linear predictor
xvals = seq(2,4)
newdata = data.frame(gpa = xvals)
eta = predict(fit, newdata = newdata, type = "link")
eta

### get estimated mean
par(mfrow = c(1,2))
mu = predict(fit, newdata = newdata, type = "response")
plot(xvals, eta, main = "Linear Predictor", xlab = "gpa", ylab = expression(eta), type = "l")
plot(xvals, mu, main = "Mean Response as a Function of the Predictor", xlab = "gpa", ylab = expression(mu), ylim = c(0,1), type = "l", lwd = 3)
