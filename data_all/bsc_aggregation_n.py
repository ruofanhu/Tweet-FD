#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import ast
import json
import re
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
from itertools import combinations
import random
from seqeval.metrics import f1_score,classification_report,accuracy_score
from sklearn.metrics import f1_score as bi_f1_score
from sklearn.metrics import classification_report as bi_classification_report
import traceback
import html      #for unescape & < >
from scipy import stats
from collections import defaultdict
import emoji
import os


# In[474]:



data_200=pd.read_csv('filtered_initial_batches.csv')
data_3000=pd.read_csv('filtered_keep_main_batches.csv')

dj_3000=data_3000.to_json(orient="records")
parsed_3000 = json.loads(dj_3000)

dj_200=data_200.to_json(orient="records")
parsed_200 = json.loads(dj_200)


# In[475]:





# In[476]:



def process_entity_label_200(record):
    txt = html.unescape(record['Input.tweet'])
    
    temp = list(filter(None, re.split('([,.!?:()[\]\\/"“”\s+])', txt)))

    # remove space strings from list and convert into np array
    tweet_split = np.array(list(filter(str.strip, temp)))


    token_labels = np.array(['O']*len(tweet_split),dtype=np.dtype(('U',10)))

                
    if record['Answer.no-entity'] is None and re.split('[|]', record["Answer.html_output"])[1]!='': # the value is 1 when there is no entity to label
        html_output_list = ast.literal_eval(re.split('[|]', record["Answer.html_output"])[1])
                
        for e in html_output_list:
            if 'idx' in list(e.keys()):

                if ' ' in e['idx']:
                    idx = list(map(int, e['idx'].split(' ')))
                else:
                    idx = ast.literal_eval(e['idx'])

                if type(idx) is int:

#                             assert tweet_split[idx] == e['text']
                    token_labels[idx] = 'B-'+e['className'].split('-')[1]
                else:
                    idx=list(idx)
                    token_labels[idx[0]] = 'B-'+e['className'].split('-')[1]
                    token_labels[idx[1:]] = 'I-' + e['className'].split('-')[1]

    return token_labels.tolist()

def process_entity_label_3000(record):
    txt = html.unescape(record['Input.tweet'])
    
    temp = list(filter(None, re.split('([,.!?:()[\]"\s+])', txt)))

    # remove space strings from list and convert into np array
    tweet_split = np.array(list(filter(str.strip, temp)))


    token_labels = np.array(['O']*len(tweet_split),dtype=np.dtype(('U',10)))
#     if record['Answer.related_index'] != '[]' :
#         relation_lables_idx_str = sum([i.split(' ') for i in ast.literal_eval(record['Answer.related_index'])],[])

#         relation_lables_idx = list(map(int, relation_lables_idx_str))
#         relation_lables[relation_lables_idx] = 1

                
    if record['Answer.no-entity'] is None and re.split('[|]', record["Answer.html_output"])[1]!='': # the value is 1 when there is no entity to label
        html_output_list = ast.literal_eval(re.split('[|]', record["Answer.html_output"])[1])
                
        for e in html_output_list:
            if 'idx' in list(e.keys()):

                if ' ' in e['idx']:
                    idx = list(map(int, e['idx'].split(' ')))
                else:
                    idx = ast.literal_eval(e['idx'])

                if type(idx) is int:

#                             assert tweet_split[idx] == e['text']
                    token_labels[idx] = 'B-'+e['className'].split('-')[1]
                else:
                    idx=list(idx)
                    token_labels[idx[0]] = 'B-'+e['className'].split('-')[1]
                    token_labels[idx[1:]] = 'I-' + e['className'].split('-')[1]

    return token_labels.tolist()


# In[477]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
reviewsPertweet = defaultdict(list)
for d in parsed_3000:
    entity_labels=process_entity_label_3000(d)
    reviewsPertweet[d['InputId']].append([1]+[0]*(len(entity_labels)-1))


    for i in range(len(entity_labels)):
        user,item = d['WorkerId'], str(d['InputId'])+','+str(i)
        reviewsPerUser[user].append(d)
        reviewsPerItem[item].append({'WorkerId':d['WorkerId'],'entity_type':entity_labels[i]})

for d in parsed_200:
    entity_labels=process_entity_label_200(d)
    reviewsPertweet[d['InputId']].append([1]+[0]*(len(entity_labels)-1))


    for i in range(len(entity_labels)):
        user,item = d['WorkerId'], str(d['InputId'])+','+str(i)
        reviewsPerUser[user].append(d)
        reviewsPerItem[item].append({'WorkerId':d['WorkerId'],'entity_type':entity_labels[i]})

        
lu = len(reviewsPerUser)
li = len(reviewsPerItem)
data_ = np.empty((li, lu))
data_[:] = np.nan
# data__t = np.empty((lu, li))
ku = list(reviewsPerUser.keys())
ki = list(reviewsPerItem.keys())


# Construct the P Matrix
data_m = pd.DataFrame('',columns=ku,index=ki)
for i in range(li):
    for r in reviewsPerItem[ki[i]]:
        data_m.loc[ki[i]][ku.index(r['WorkerId'])] =r['entity_type']






list1=[i.split(',')[0] for i in list(data_m.index)]
all_tweet_id = []
 
for i in list1:
    if i not in all_tweet_id:
        all_tweet_id.append(i)
tokens_count=[list1.count(i) for i in all_tweet_id]




label2num={'O':8,'':-1,'B-food':2, 'B-loc':0, 'B-other':6, 'B-symptom':4,'I-food':3, 'I-loc':1,
       'I-other':7, 'I-symptom':5}

num2label={v:k for k, v in label2num.items()}





#annotations
new_data_m=data_m.replace(label2num)





## Create doc_start and features
ppdict = {n: grp.loc[n].to_dict('index') for n, grp in data_3000.set_index(['Input.id', 'WorkerId']).groupby(level='Input.id')}

doc_start=[]
features=[]
for tweet_id,collection in ppdict.items():
    basic_info=list(collection.values())[0]
    txt = html.unescape(basic_info['Input.tweet'])
    temp = list(filter(None, re.split('([,.!?:()[\]"\s+])', txt)))

    # remove space strings from list and convert into np array
    tweet_split = list(filter(str.strip, temp))
    length=len(tweet_split)
    doc_start.extend([1]+[0]*(length-1))
    features.extend(tweet_split)

ppdict = {n: grp.loc[n].to_dict('index') for n, grp in data_200.set_index(['Input.id', 'WorkerId']).groupby(level='Input.id')}

for tweet_id,collection in ppdict.items():
    basic_info=list(collection.values())[0]
    txt = html.unescape(basic_info['Input.tweet'])
    temp = list(filter(None, re.split('([,.!?:()[\]\\/"“”\s+])', txt)))

    # remove space strings from list and convert into np array
    tweet_split = list(filter(str.strip, temp))
    length=len(tweet_split)
    doc_start.extend([1]+[0]*(length-1))
    features.extend(tweet_split)

    
ll=list(reviewsPertweet.values())
doc_start=sum([i[0] for i in ll],[])
features=np.array(features)



from bayesian_combination import bayesian_combination



print(new_data_m.shape)




num_classes = 9 # Beginning, Inside and Outside

bc_model = bayesian_combination.BC(L=num_classes, K=2515, annotator_model='seq', max_iter=20,tagging_scheme='IOB2',inside_labels=[1,3,5,7], outside_label=8, beginning_labels=[0,2,4,6])
values= bc_model.fit_predict(new_data_m.to_numpy(), np.array(doc_start), features)



prob=values[0]

agg=values[1]



agg_predictions=[]
for i in tokens_count:
    agg_predictions.append(agg[:i].tolist())
    agg=np.delete(agg, range(i))



final_agg=[list((pd.Series(agg_p)).map(num2label).values) for agg_p in agg_predictions]



np.save('entity_agreegation_new.npy', np.array(final_agg), allow_pickle=True)

