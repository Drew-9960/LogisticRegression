# stepwise Regression for feature selection (most significant variables)

insurance = read.csv('insurance.csv')

str(insurance)

ins_model = lm(charges~age + children + bmi + factor(sex) + 
                 factor(smoker) + factor(region), data = insurance)
ins_model
summary(ins_model)

stepwise_mod = step(ins_model)
summary(stepwise_mod)

str(insurance)
insurance$sex = NULL

# Checking for Multicollinarity, Variation Inflation Factor (VIF)
# install.packages('car')
library(car)

vif(stepwise_mod)

# Relative importance of variables (Ranking)
# install.packages('relaimpo')
library(relaimpo)

importance_stepmod = calc.relimp(stepwise_mod, type = "lmg", rela = TRUE) #lmg == linear regression # LOOK INTO
importance_stepmod

sort(importance_stepmod$lmg, decreasing = TRUE)


# Logistic Regression 

inputdata = read.csv('adult.csv')
str(inputdata)

table(inputdata$income)

inputdata$above50k = ifelse(inputdata$income == " <=50K", 0, 1) #dummy variable creation
table(inputdata$above50k)
barplot(table(inputdata$above50k))

# Using stratified sampling in training an test sets to keep the dependent variable in equal proportions
# install.packages('splitstackshape')

library(splitstackshape)

set.seed(1000)

splitdata = stratified(inputdata, "above50k", 0.8, bothSets = TRUE)
trainingdata = as.data.frame(splitdata$SAMP1)
testdata = as.data.frame(splitdata$SAMP2)


### checking proportions of split data 
table(trainingdata$above50k)
table(testdata$above50k)
sum(trainingdata$above50k==1)/sum(trainingdata$above50k==0)
sum(testdata$above50k==1)/sum(testdata$above50k==0)

# logistic model 
str(trainingdata)

logitmod = glm(above50k ~ relationship + marital.status + age + capital.gain + occupation + education.num,
               data = trainingdata)

predicted = predict(logitmod, testdata)
head(predicted, 10)

View(testdata)

# Accuracy, Sensitivity, Specificity 
# install.packages('caret')
# install.packages('e1071')
library(caret)

predicted_factor = ifelse(predicted >= 0.5, 1, 0)
confusionMatrix(as.factor(testdata$above50k), as.factor(predicted_factor), positive = '1')

confusionMatrix(as.factor(testdata$above50k), as.factor(predicted_factor), positive = '1')

# AUC : Area under the curve
# install.packages('ROCR')
library(ROCR)

# AUC Measure
pred_df = prediction(predicted, testdata$above50k)
auc_value = performance(pred_df, measure = 'auc')

auc_value@y.values

# ROC Curve

roc_curve = performance(pred_df, 'tpr', 'fpr')
plot(roc_curve)


