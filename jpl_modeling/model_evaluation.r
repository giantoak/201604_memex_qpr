library(gbm)
library(caret)
library(ROCR)

#After running python script to add in actual classes to eval data, load it back in to generate ROC curve
final = read.csv("/users/hundman/documents/data_science/memex-qpr/eval_predictions_with_class.csv", header=T)

# ROC curve analysis
ROCRpred<-prediction(eval_data_phone$score,final$match)
performance(ROCRpred,"auc")
performance(ROCRpred,"prbe")
plot(performance(ROCRpred, 'tpr', 'fpr'))
abline(a=0, b= 1)

#Confusion matrix
predicted = final$score > .79 #use "prbe" value from above
confusionMatrix(predicted, final$match)
# confusionMatrix(yhatTrain, test$match)