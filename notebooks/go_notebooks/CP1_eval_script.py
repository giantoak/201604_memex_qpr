#!/usr/bin/env python

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import sys
import json
import ipdb
import pandas as pd

# how to use: python roc-test.py ht_evaluation_UPDATED your_group-s_cool_data_file.txt
#
################################################
# do not edit - eval data
eval_id = []
eval_phones = []
eval_scores = []
eval_outputs = open(sys.argv[1], "r")
for line in eval_outputs:
    eval_id.append(json.loads(line)['ad']['_id'])
    eval_phones.append(json.loads(line)["phone"])
    score = json.loads(line)['class']
    if score == 'positive':
        score = 1
    elif score == 'negative':
        score = 0
    else:
        print "TROUBLE"
        break
    eval_scores.append(score)
eval_outputs.close()
eval_data_id_level = pd.DataFrame({'cdr_id':eval_id, 'score':eval_scores, 'phone':[i[0] for i in eval_phones]})
eval_data_phone_level = eval_data_id_level.groupby('phone')['score'].max().reset_index()

################################################

################################################
# group data ingest - edit to fit your data as needed
# ad level
ad_predictions = pd.DataFrame(json.loads(open(sys.argv[2], "r").read()))
ad_predictions = ad_predictions.rename(columns={'score':'predicted'})
ad_predictions = ad_predictions.merge(eval_data_id_level, how='left')
print('missing evaluation data for %s ads' % ad_predictions.score.isnull().sum())
print(ad_predictions.shape)
ad_predictions = ad_predictions[ad_predictions['score'].notnull()]
print(ad_predictions.shape)

# phone level
phone_predictions = pd.DataFrame(json.loads(open(sys.argv[3], "r").read()))
phone_predictions = phone_predictions.rename(columns={'score':'predicted'})
phone_predictions['phone'] = phone_predictions['phone'].apply(lambda x: x.replace('(','').replace(')','').replace(' ','').replace('-',''))
phone_predictions = phone_predictions.merge(eval_data_phone_level, how='left')
print('missing evaluation data for %s phones' % phone_predictions.score.isnull().sum())
print(phone_predictions.shape)
phone_predictions = phone_predictions[phone_predictions['score'].notnull()]
print(phone_predictions.shape)
################################################

fpr ,tpr, thresholds = roc_curve(phone_predictions['score'], phone_predictions['predicted'])
auc = roc_auc_score(phone_predictions['score'], phone_predictions['predicted'])
print 'ROC-AUC is:', auc
print 'ROC curve plotting'
plt.plot(fpr, tpr, '.-')
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.savefig('phone_level.pdf')
#plt.show()

fpr ,tpr, thresholds = roc_curve(ad_predictions['score'], ad_predictions['predicted'])
auc = roc_auc_score(ad_predictions['score'], ad_predictions['predicted'])
print 'ROC-AUC is:', auc
print 'ROC curve plotting'
plt.plot(fpr, tpr, '.-')
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
#plt.show()
plt.savefig('ad_level.pdf')
