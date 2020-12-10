#R version 4.0.3 (2020-10-10)
#Date: 28th November 2020
#Author: Eleanor Barnes
#UCI Processed Cleveland Dataset

#Supervised statistical learning
#to identify the combination of features that are more likely to be associated with heart disease.

library(tidyverse)
library(xgboost)
library(caret)
library(Hmisc)
library(e1071)
library(glmnet)
library(pROC)


set.seed(100)

###Functions-----
normalise = function(x){
  normalised = (x - min(x))/(max(x)-min(x))
}

xgb_cv = function(train_data, train_label, k = 5){

grid_cv <- expand.grid( eta = c(0.01, 0.001, 0.0001),
                        max_depth = c(4,6,8,10), 
                        gamma=c(0:5), 
                        subsample=seq(0.5,1,by = 0.1)
)
best_params = data.frame(eta = NA, max_depth = NA, gamma = NA, subsample = NA, accuracy =NA, k = NA)
for(n in 1:k){
  set.seed(n*1000)
  train_index= sample(nrow(train_data),floor(0.75*nrow(train_data)))
  train_data = as.matrix(train_data[train_index,])
  validation_data = as.matrix(train_data[-train_index,])
  
  train_label = train_label[train_index]
  validation_label = train_label[-train_index]
  
  #
  xgb_train = xgb.DMatrix(data=train_data,label=train_label)

  
  all_accuracy = vector(length = nrow(grid_cv))
  
  for(i in 1:nrow(grid_cv)){
  params = list(
    booster="gbtree",
    eta = grid_cv$eta[i],
    max_depth = grid_cv$max_depth[i],
    gamma = grid_cv$gamma[i],
    subsample = grid_cv$subsample[i],
    objective ="binary:logistic",
    eval_metric ="logloss"
  )
  
  
  xgb_fit=xgb.train(
    params=params,
    data=xgb_train,
    nrounds=10000,
    early_stopping_rounds=10,
    watchlist=list(val1=xgb_train,val2=xgb_test),
    verbose=0
  )

  #Evaluating model
  
  xgb_validation_predictions = round(predict(xgb_fit,validation_data,reshape=T))
  xgb_validation_accuracy = sum(xgb_validation_predictions==validation_label)/length(xgb_validation_predictions)
  print(as.character(i))
  all_accuracy[i] = xgb_validation_accuracy
  }
  best_params[n,] = cbind(grid_cv[which.max(all_accuracy),], accuracy = max(all_accuracy), k = n)
  
}

  return(best_params)
}


#Load heart disease dataset---
raw_data = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header = F)
str(raw_data) #Ensuring all rows/columns have loaded correctly
colnames(raw_data) = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca","thal", "outcome")
raw_data$ca = as.numeric(raw_data$ca)
raw_data$thal = as.numeric(raw_data$thal)

###Checking for odd structures -----
sum(is.na(raw_data)) #NAs introduced due to numeric conversion, they are missing values
anyDuplicated(raw_data)

data = na.omit(raw_data) #Omitted rows with missing data


###Exploring the dataset----

ggplot(pivot_longer(data, 1:13, names_to = "metric", values_to = "value"), aes(x = value)) + 
  geom_histogram()+ 
  facet_wrap(~metric, scales = "free")+
  labs(title = 'Distribution of features')

summary(data)

###XG boost: Binary outcome ----

#Data preprocessing
binary_data = data%>%
  mutate(outcome = case_when(outcome>0 ~ 1, TRUE ~ 0))%>% # Turning outcome into binary - either has heart disease or doesn't
  mutate(thal = as.character(thal), 
         cp =as.character(cp),
         slope = as.character(slope) 
         )

#One-hot encoding all categorical predictor variables. Else, XGBoost preserves numeric values 
dummification = dummyVars(outcome ~., binary_data)
dummy_data = data.frame(predict(dummification, binary_data))

label = as.integer(binary_data$outcome)

#Train-test split
train_index= sample(nrow(dummy_data),floor(0.75*nrow(dummy_data)))
train_data = as.matrix(dummy_data[train_index,])
test_data = as.matrix(dummy_data[-train_index,])

train_label = label[train_index]
test_label = label[-train_index]

#
xgb_train = xgb.DMatrix(data=train_data,label=train_label)
xgb_test = xgb.DMatrix(data=test_data,label=test_label)

#Cross validating hyperparameters
highest_performers = xgb_cv(train_data, train_label, k=1) %>%
  group_by(eta, max_depth, gamma, subsample)%>%
  summarise(mean = mean(accuracy, na.rm), count = n()) %>%
  filter(max(count))
cv_params = list(
  booster="gbtree",
  eta=highest_performers$eta,
  max_depth=highest_performers$max_depth,
  gamma=highest_performers$gamma,
  subsample=highest_performers$subsample,
  colsample_bytree=1,
  objective="binary:logistic",
  eval_metric="logloss"
)

from_internet_params = list(
  booster="gbtree",
  eta=0.1,
  max_depth=4,
  gamma=3,
  subsample=0.5,
  colsample_bytree=1,
  objective="binary:logistic",
  eval_metric="logloss"
)

#Train model
binary_xgb_fit=xgb.train(
  params=from_internet_params,
  data=xgb_train,
  nrounds=10000,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb_train,val2=xgb_test),
  verbose=0
)

#Plotting feature importance
binary_feature_importance <- xgb.importance (feature_names = colnames(dummy_data), model = binary_xgb_fit)
xgb.plot.importance(importance_matrix = binary_feature_importance) 

#Evaluating model
binary_xgb_train_predictions = round(predict(binary_xgb_fit,train_data,reshape=T))
binary_xgb_train_accuracy = caret::confusionMatrix(as.factor(binary_xgb_train_predictions), as.factor(train_label))

binary_xgb_test_predictions = round(predict(binary_xgb_fit,test_data,reshape=T))
binary_xgb_test_accuracy = caret::confusionMatrix(as.factor(binary_xgb_test_predictions), as.factor(test_label))

###Logistic regression----
lr_data = data%>%
  mutate(outcome = case_when(outcome>0 ~ 1, TRUE ~ 0))

#Train-test split, Normalising data for lasso regularisation
train_index= sample(nrow(lr_data),floor(0.75*nrow(lr_data)))
train_data = lr_data[train_index,] %>% mutate(age = normalise(age), 
                                                 trestbps = normalise(trestbps), 
                                                 chol = normalise(chol), 
                                                 thalach = normalise(thalach), 
                                                 oldpeak = normalise(oldpeak), 
                                                 ca = normalise(ca))
test_data = lr_data[-train_index,] %>% mutate(age = normalise(age), 
                                                 trestbps = normalise(trestbps), 
                                                 chol = normalise(chol), 
                                                 thalach = normalise(thalach), 
                                                 oldpeak = normalise(oldpeak), 
                                                 ca = normalise(ca))


#Running a full predictive model
lr_full_model = glm(outcome ~., data = train_data, family = "binomial"(link ='logit')) 
summary(lr_full_model)

anova(lr_full_model, test = 'Chisq')
null = glm(outcome ~1, data = train_data, family = "binomial"(link ='logit')) 
step = MASS::stepAIC(null, scope=list(lower=.~1, upper = formula(lr_full_model)), direction = 'both')

lr_step_model = glm(step$formula, data = train_data, family = "binomial"(link ='logit')) 

#Evaluating models
lr_predictions_train =  round(predict(lr_full_model, newdata = train_data, type = 'response'))
lr_predictions_test =  round(predict(lr_full_model, newdata = test_data, type = 'response'))

lr_accuracy = caret::confusionMatrix(as.factor(as.numeric(lr_predictions_test)), as.factor(test_data$outcome))

#Lasso regularisation
lasso_train_data = as.matrix(select(train_data, -outcome))
lasso_train_label = as.matrix(train_data$outcome)

lasso_test_data = as.matrix(select(test_data, -outcome))
lasso_test_label = as.matrix(test_data$outcome)


#Cross validate for lambda value
lambdas = 10^seq(-2,2, 0.1)
cv_lasso = cv.glmnet(lasso_train_data, lasso_train_label, alpha = 1, family = binomial(link = 'logit'), lambda = lambdas)

#Train model
lr_lasso_model = glmnet(lasso_train_data, lasso_train_label, alpha = 1, family = binomial(link = 'logit'), lambda = cv_lasso$lambda.min)

#Coefficients of features
coefficients = as.data.frame(as.matrix(coef(lr_lasso_model)))%>%
  arrange(s0)


#Evaluate model
train_predictions = round(predict(lr_lasso_model, lasso_train_data, s = cv_lasso$lambda.min, type = 'response'))
test_predictions = round(predict(lr_lasso_model, lasso_test_data, s = cv_lasso$lambda.min, type = 'response'))

caret::confusionMatrix(as.factor(as.numeric(train_predictions)), as.factor(lasso_train_label))
lasso_accuracy = caret::confusionMatrix(as.factor(as.numeric(test_predictions)), as.factor(lasso_test_label))


###Best performing model-----

print(paste0('Accuracy on test data...Lasso:', as.character(lasso_accuracy$overall[1]), ' Logistic Regression:', as.character(lr_accuracy$overall[1])))