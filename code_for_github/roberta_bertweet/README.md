# RoBERTa and BERTweet
Codes for training RoBERTa and BERTweet on TWEET-FD dataset.

## Usage
### Train TRC model
```linux
python main_sequence.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --model_type bertweet-seq \
   --data /tweet-fd \
   --data_file tweet_fd.p \
   --assign_weight \
   --n_epochs 20 \
   --log_dir test \
```
### Train EMD/ERC/RED model
```linux
python main_token.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --task_type [entity_detection|entity_relevance_classification|relevant_entity_detection] \
   --model_type [bertweet-token|bertweet-token-crf] \
   --data /tweet-fd \
   --data_file tweet_fd.p \
   --label_map label_map.json \
   --assign_weight \
   --n_epochs 20 \
   --log_dir test \
```
### Train TRC+EMD/ERC/RED model
```linux
python main_multi.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --model_type [bertweet-multi|bertweet-multi-crf] \
   --task_type [entity_detection|entity_relevance_classification|relevant_entity_detection] \
   --data /tweet-fd \
   --data_file tweet_fd.p \
   --label_map label_map.json \
   --assign_token_weight \
   --assign_seq_weight \
   --n_epochs 20 \
   --token_lambda 10 \
   --log_dir test \
```
### Train EMD+ERC model
```linux
python main_two_token.py \
   --seed 2021 \
   --bert_model [roberta-base|vinai/bertweet-base] \
   --task_type 'entity_detection & entity_relevance_classification' \
   --model_type [bertweet-two-token|bertweet-two-token-crf] \
   --data /tweet-fd \
   --data_file tweet_fd.p \
   --label_map label_map.json \
   --se_label_map re_label_map.json \
   --assign_token_weight \
   --assign_se_weight \
   --n_epochs 20 \
   --log_dir test \
```
