# RoBERTa, BERTweet, BiLSTM, and SVM
Codes for training RoBERTa, BERTweet, BiLSTM, and SVM on TWEET-FD dataset.
Below and default parameters in each code files are parameter settings that we used to train models in paper's experiment section.

## Usage
### Train TRC model
#### BERTweet or RoBERTa
```linux
python main_sequence.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --model_type bertweet-seq \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --assign_weight \
   --learning_rate 1e-5 \
   --n_epochs 20 \
   --log_dir test \
```

#### BiLSTM
```linux
python main_sequence.py \
   --seed 2021 \
   --model_type BiLSTM-seq \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --assign_weight \
   --n_epochs 20 \
   --learning_rate 1e-3 \
   --log_dir test \
   --embeddings_file glove.840B.300d.txt \
```
assign_weight is to determine if assign weights to cross entropy loss, weight is calculated by the proportion of tweet classes in training set

n_epochs is number of epochs to training model

#### SVM
see [Bulid_SVM_model.ipynb](Bulid_SVM_model.ipynb)


### Train EMD/ERC/RED model
#### BERTweet or RoBERTa
```linux
python main_token.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --task_type [entity_detection|entity_relevance_classification|relevant_entity_detection] \
   --model_type [bertweet-token|bertweet-token-crf] \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --label_map [label_map.json|re_label_map.json] \
   --assign_weight \
   --n_epochs 20 \
   --learning_rate 1e-5 \
   --log_dir test \
```

#### BiLSTM
```linux
python main_token.py \
   --seed 2021 \
   --task_type [entity_detection|entity_relevance_classification|relevant_entity_detection] \
   --model_type [BiLSTM-token|BiLSTM-token-crf] \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --label_map [label_map.json|re_label_map.json] \
   --assign_weight \
   --n_epochs 20 \
   --learning_rate 1e-3 \
   --log_dir test \ 
   --embeddings_file glove.840B.300d.txt \
```
label_map.json is the label_map for EMD and RED task, re_label_map.json is for ERC task.

assign_weight is to determine if assign weight to cross entropy loss, weight is calculated by the proportion of entities' frequeny in training set


### Train TRC+EMD/ERC/RED model
#### BERTweet or RoBERTa
```linux
python main_multi.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --model_type [bertweet-multi|bertweet-multi-crf] \
   --task_type [entity_detection|entity_relevance_classification|relevant_entity_detection] \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --label_map [label_map.json|re_label_map.json] \
   --assign_token_weight \
   --assign_seq_weight \
   --n_epochs 20 \
   --learning_rate 1e-5 \
   --token_lambda 10 \
   --log_dir test \
```
#### BiLSTM
```linux
python main_multi.py \
   --seed 2021 \
   --model_type [BiLSTM-multi|BiLSTM-multi-crf] \
   --task_type [entity_detection|entity_relevance_classification|relevant_entity_detection] \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --label_map [label_map.json|re_label_map.json] \
   --assign_token_weight \
   --assign_seq_weight \
   --n_epochs 20 \
   --learning_rate 1e-3 \
   --token_lambda 10 \
   --log_dir test \
   --embeddings_file glove.840B.300d.txt \
```
token_lambda is the lambda value assiging to word level task loss, we used 10 in our experiment.

assign_token_weight is to determine if assign weight to word level task cross entropy loss, weight is calculated by the proportion of entities' frequeny in training set

assign_seq_weight is to determine if assign weights to TRC task cross entropy loss, weight is calculated by the proportion of tweet classes in training set

### Train EMD+ERC model
#### BERTweet or RoBERTa
```linux
python main_two_token.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --task_type 'entity_detection & entity_relevance_classification' \
   --model_type [bertweet-two-token|bertweet-two-token-crf] \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --label_map label_map.json \
   --se_label_map re_label_map.json \
   --assign_token_weight \
   --assign_se_weight \
   --n_epochs 20 \
   --learning_rate 1e-5 \
   --log_dir test \
```

#### BiLSTM
```linux
python main_two_token.py \
   --seed 2021 \
   --task_type 'entity_detection & entity_relevance_classification' \
   --model_type [BiLSTM-two-token|BiLSTM-two-token-crf] \
   --data /tweet-fd \
   --train_file train.p \
   --val_file val.p \
   --test_file test.p \
   --label_map label_map.json \
   --se_label_map re_label_map.json \
   --assign_token_weight \
   --assign_se_weight \
   --n_epochs 20 \
   --learning_rate 1e-3 \
   --log_dir test \
   --embeddings_file glove.840B.300d.txt \
```

assign_token_weight is to determine if assign weight to EMD task cross entropy loss, weight is calculated by the proportion of entities' frequeny in training set

assign_se_weight is to determine if assign weights to ERC task cross entropy loss, weight is calculated by the proportion of entities' in training set
