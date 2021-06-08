import re
import json
import random
from abc import ABCMeta
from pathlib import Path
from typing import List, Tuple, Dict
from itertools import zip_longest
import fasttext
from more_itertools import flatten
import numpy as np
import pandas as pd
import tensorflow as tf
from IMGJM.utils import pad_char_sequences
from sklearn.model_selection import train_test_split


class BaseDataset(metaclass=ABCMeta):
    def __init__(self,
                 data_dir: str = None,
                 resource: str = None,
                 char2id: Dict = None,
                 word2id: Dict = None,
                 shuffle: bool = True,
                 seed: int = 1,
                 *args,
                 **kwargs):
        self.base_data_dir = Path('dataset')
        if resource:
            self.train_data_fp = self.base_data_dir / f'{data_dir}_{resource}' / 'train.txt'
            self.dev_data_fp = self.base_data_dir / f'{data_dir}_{resource}' / 'dev.txt'
            self.test_data_fp = self.base_data_dir / f'{data_dir}_{resource}' / 'test.txt'
            self.label_map_fp = self.base_data_dir / f'{data_dir}_{resource}' / 'label_map.json'
            self.re_label_map_fp = self.base_data_dir / f'{data_dir}_{resource}' / 're_label_map.json'
        else:
            self.train_data_fp = self.base_data_dir / data_dir / 'train.txt'
            self.dev_data_fp = self.base_data_dir / data_dir / 'dev.txt'
            self.test_data_fp = self.base_data_dir / data_dir / 'test.txt'
            self.label_map_fp = self.base_data_dir / data_dir / 'label_map.json'
            self.re_label_map_fp = self.base_data_dir / data_dir / 're_label_map.json'
        
        self.raw_train_data = BaseDataset.read_txtfile_tolist(self.train_data_fp) 
        self.raw_dev_data = BaseDataset.read_txtfile_tolist(self.dev_data_fp)
        self.raw_test_data = BaseDataset.read_txtfile_tolist(self.test_data_fp) 
        with open(self.label_map_fp, 'r') as fp:
            self.label_map = json.load(fp)
        with open(self.re_label_map_fp, 'r') as fp:
            self.re_label_map = json.load(fp)
        self.char2id = char2id
        self.word2id = word2id
        if shuffle:
            self.raw_train_data = BaseDataset.shuffle_all(self.raw_train_data)
            self.raw_dev_data = BaseDataset.shuffle_all(self.raw_dev_data)
            self.raw_test_data = BaseDataset.shuffle_all(self.raw_test_data)

    @staticmethod
    def shuffle_all(data):
        text, label, related_label = data
        c = list(zip(text, label, related_label))
        random.shuffle(c)
        text, label, related_label = zip(*c)
        return text, label, related_label
    
    @property
    def word2id(self):
        return self._word2id

    @word2id.setter
    def word2id(self, w2i):
        if w2i:
            self._word2id = w2i
        else:
            self._word2id = self.build_word2id(self.raw_train_data[0] + self.raw_dev_data[0] +
                                               self.raw_test_data[0])

    @property
    def char2id(self):
        return self._char2id

    @char2id.setter
    def char2id(self, c2i):
        if c2i:
            self._char2id = c2i
        else:
            self._char2id = self.build_char2id(self.raw_train_data[0] + self.raw_dev_data[0] +
                                               self.raw_test_data[0])

    @staticmethod
    def build_word2id(sentence_list: List) -> Dict:
        word2id = {}
        sentences = [sent
            for sent in sentence_list
        ]
        sentences_flatten = sorted(list(set(flatten(sentences))))
        for i, word in enumerate(sentences_flatten):
            word2id[word] = i
        return word2id

    @staticmethod
    def build_char2id(sentence_list: List) -> Dict:
        char2id = {}
        sentences = [
            ' '.join(sent) for sent in sentence_list
        ]
        char_list = sorted(list(set(''.join(sentences))))
        for i, char in enumerate(char_list):
            char2id[char] = i
        return char2id

    
    @staticmethod
    def extract_from_dataframe(dataframe, columns, index):
        return_list = []
        for col in columns:
            return_list.append(dataframe[col].loc[index].to_numpy().tolist())
        return return_list
    
    @staticmethod
    def read_txtfile_tolist(filepath):
        all_text = []
        all_labels = []
        all_related_labels = []
        orig_tokens = []
        orig_labels = []
        orig_related_labels = []

        file = open(filepath, "rt")
        d = file.readlines()
        file.close()
        for line in d:
            line = line.rstrip()

            if not line:
                all_text.append(orig_tokens)
                all_labels.append(orig_labels)
                all_related_labels.append(orig_related_labels)
                orig_tokens = []
                orig_labels = []
                orig_related_labels = []
            else:
                token, label, related_label = line.split('\t')
                orig_tokens.append(token)
                orig_labels.append(label)
                orig_related_labels.append(related_label)
        return all_text, all_labels, all_related_labels
    

    @staticmethod
    def format_dataset(all_data: Tuple[str], label_map:dict, re_label_map:dict) -> Tuple:
        all_text, all_labels, all_related_labels = all_data
        all_labels_ = [[label_map[i] for i in ll] for ll in all_labels]
        all_related_labels_ = [[re_label_map[i] for i in ll] for ll in all_related_labels]
        return all_text, all_labels_, all_related_labels_

    def transform(self, sentence_list: List[List]
                  ) -> Tuple[List[List[List]], List[List]]:
        '''
        Transform segmented sentence into char & word ids

        Args:
            sentence (list)

        Returns:
            char_ids (list)
            word_ids (list)
        '''
        char_ids, word_ids = [], []
        for sent in sentence_list:
            wids, cids = [], []
            for word in sent:
                wids.append(self.word2id.get(word, len(self.word2id) + 1))
                cid_in_cid = []
                for char in word:
                    cid_in_cid.append(
                        self.char2id.get(char,
                                         len(self.char2id) + 1))
                cids.append(cid_in_cid)
            word_ids.append(wids)
            char_ids.append(cids)
        return char_ids, word_ids

    def merge_all(self, sentences: List[str]) -> Tuple[List]:
        sents, entities, polarities = self.format_dataset(sentences,self.label_map,self.re_label_map)
        char_ids, word_ids = self.transform(sents)
        sequence_length = [len(sent) for sent in word_ids]
        return char_ids, word_ids, sequence_length, entities, polarities

    def merge_and_pad_all(self, sentences: List[str]) -> Tuple[np.ndarray]:
        sents, entities, polarities = self.format_dataset(sentences,self.label_map,self.re_label_map)
        char_ids, word_ids = self.transform(sents)
        pad_char_ids = pad_char_sequences(char_ids,
                                          padding='post',
                                          value=len(self.char2id))
        pad_word_ids = tf.keras.preprocessing.sequence.pad_sequences(
            word_ids, padding='post', value=len(self.word2id))
        sequence_length = [len(sent) for sent in word_ids]
        pad_entities = tf.keras.preprocessing.sequence.pad_sequences(
            entities, padding='post', value=0)
        pad_polarities = tf.keras.preprocessing.sequence.pad_sequences(
            polarities, padding='post', value=0)
        return pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities

    @property
    def train_data(self) -> Tuple[List]:
        outputs = self.merge_all(self.raw_train_data)
        return outputs
    
    @property
    def dev_data(self) -> Tuple[List]:
        outputs = self.merge_all(self.raw_dev_data)
        return outputs

    @property
    def test_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_all(self.raw_test_data)
        return outputs

    @property
    def pad_train_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_and_pad_all(self.raw_train_data)
        return outputs
    
    @property
    def pad_dev_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_and_pad_all(self.raw_dev_data)
        return outputs

    @property
    def pad_test_data(self) -> Tuple[np.ndarray]:
        outputs = self.merge_and_pad_all(self.raw_test_data)
        return outputs

    def batch_generator(self, batch_size: int = 32,
                        data_source: str = 'train') -> Tuple[np.ndarray]:
        if data_source == 'train':
            dataset = self.raw_train_data
        elif data_source == 'dev':
            dataset = self.raw_dev_data
        else:
            dataset = self.raw_test_data
        if data_source != 'train':
            batch_size = len(dataset)
        start, end = 0, batch_size
        batch_nums = (len(dataset) // batch_size) + 1
        for _ in range(batch_nums):
            pad_char_ids, pad_word_batch, sequence_length, pad_entities, pad_polarities = self.merge_and_pad_all(
                dataset[start:end])
            yield (pad_char_ids, pad_word_batch, sequence_length, pad_entities,
                   pad_polarities)
            start, end = end, end + batch_size
            if start >= len(dataset):
                break



class Twitter(BaseDataset):
    def __init__(self,
                 data_dir: str = 'tweet-fd',
                 char2id: Dict = None,
                 word2id: Dict = None,
                 data_file: str = 'tweet_fd.p',
                 shuffle: bool = True,
                 seed: int = 1,
                 *args,
                 **kwargs):
        self.base_data_dir = Path('/work/dzhang5/usda_project/benchmark')
        self.all_data_fp = self.base_data_dir / data_dir / data_file
        all_data = pd.read_pickle(self.all_data_fp)
        train_index, val_test_index = train_test_split(all_data.index, test_size=0.2, random_state=seed)
        val_index, test_index = train_test_split(val_test_index, test_size=0.5, random_state=seed)
        need_columns = ['tweet_tokens', 'entity_label', 'relevance_entity_class_label']
        self.raw_train_data = BaseDataset.extract_from_dataframe(all_data, need_columns, train_index)
        self.raw_dev_data = BaseDataset.extract_from_dataframe(all_data, need_columns, val_index)
        self.raw_test_data = BaseDataset.extract_from_dataframe(all_data, need_columns, test_index)
        
        self.label_map_fp = self.base_data_dir / data_dir / 'label_map.json'
        self.re_label_map_fp = self.base_data_dir / data_dir / 're_label_map.json'
        
        with open(self.label_map_fp, 'r') as fp:
            self.label_map = json.load(fp)
        with open(self.re_label_map_fp, 'r') as fp:
            self.re_label_map = json.load(fp)
        self.char2id = char2id
        self.word2id = word2id
        if shuffle:
            self.raw_train_data = BaseDataset.shuffle_all(self.raw_train_data)
            self.raw_dev_data = BaseDataset.shuffle_all(self.raw_dev_data)
            self.raw_test_data = BaseDataset.shuffle_all(self.raw_test_data)
        