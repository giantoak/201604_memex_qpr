library(gbm)
library(caret)
library(ROCR)

drops = c("phone")

train_data_phone = read.csv("/users/hundman/documents/data_science/memex-qpr/modeling_training.csv", header=T)
# drop phone number so it isn't used as feature in model
train_data = train_data_phone[ , !(names(train_data_phone) %in% drops)]

eval_data_phone = read.csv("/users/hundman/documents/data_science/memex-qpr/modeling_eval.csv", header=T)
# drop phone number so it isn't used as feature in model
eval_data = eval_data_phone[ , !(names(eval_data_phone) %in% drops)]

#Generate random split for training/test
# bound = floor((nrow(ht_data)/5)*4)  #define % of training and test set
# train_data <- train_data[sample(nrow(train_data)), ]  #sample rows 
# train <- train_data[1:bound, ]  #get training set
# test <- train_data[(bound+1):nrow(train_data), ]

#fit boosted tree
gbm_ht = gbm(match ~ ., data=train_data, n.trees=100000, shrinkage=0.00001, interaction.depth=3, 
	bag.fraction = .5, train.fraction = .9, n.minobsinnode = 2, cv.folds = 7, keep.data=TRUE, verbose = FALSE)

#predict evaluation data classes
eval_data_phone$score = predict(gbm_ht, eval_data, type='response')
# yhatTrain = predict(gbm_ht, test, type='response')

#model details
best.iter = gbm.perf(gbm_ht, method='cv'); best.iter
sqrt(gbm_ht$cv.error[best.iter])
summary(gbm_ht, n.trees=best.iter)

# marginal feature importances
plot(gbm_ht, i.var=2, n.tress=best.iter) #mm_score
plot(gbm_ht, i.var=4, n.tress=best.iter) #flags_cnt
plot(gbm_ht, i.var=66, n.tress=best.iter) #domains
plot(gbm_ht, i.var=68, n.tress=best.iter) #num_cities

#write out eval data with predictions
write.csv(eval_data_phone, file = "/users/hundman/documents/data_science/memex-qpr/eval_predictions.csv")



