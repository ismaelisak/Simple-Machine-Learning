library(caret)

data("iris")
dataset <- iris

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

dim(dataset)

#list the classes in dataset

sapply(dataset, class)

#looking through the dataset
head(dataset)

#Going through the levels of dataset classes

levels(dataset$Species)

#Summarize the class distribution

percentage <- prop.table(table(dataset$Species))*100
cbind(freq=table(dataset$Species), percentage=percentage)

#Summarize the attributes of the distributions

summary(dataset)

#Visualize the dataset through univariate and multivariate plots

x <- dataset[,1:4]
y <- dataset[,5]

par(mfrow=c(1,4))
for(i in 1:4){
  boxplot(x[,i], main=names(iris)[i])
}


#class breakdown for dataset

plot(y)

#multivariate plots

featurePlot(x,y,plot = "ellipse")
featurePlot(x,y, plot = "box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#Evaluating algorithms and building models

#Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)

# 1) linear algorithms
fit.lda <- train(Species~., data=dataset, method="lda", metric="Accuracy", trControl=control)

# 2) nonlinear algorithms
# CART
fit.cart <- train(Species~., data=dataset, method="rpart", metric="Accuracy", trControl=control)
# kNN
fit.knn <- train(Species~., data=dataset, method="knn", metric="Accuracy", trControl=control)
# 3) advanced algorithms
# SVM
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric="Accuracy", trControl=control)
# Random Forest
fit.rf <- train(Species~., data=dataset, method="rf", metric="Accuracy", trControl=control)


#Selecting the best model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

#Making predictions using the best model
# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
as.table(confusionMatrix(predictions, validation$Species))

str(confusionMatrix(predictions, validation$Species))
