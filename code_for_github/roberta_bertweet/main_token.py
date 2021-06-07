#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from utils import *
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import argparse
import datetime
import logging
import os, os.path
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from seqeval.metrics import accuracy_score
from transformers import AdamW, AutoTokenizer, AutoConfig, RobertaConfig
from model_weighted_roberta import *
import json


def simple_tokenize(orig_tokens, tokenizer, orig_labels, label_map, max_seq_length):
    """
    tokenize a array of raw text
    """
    # orig_tokens = orig_tokens.split()

    pad_token_label_id = -100
    tokens = []
    label_ids = []
    for word, label in zip(orig_tokens, orig_labels):
        word_tokens = tokenizer.tokenize(word)

        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    bert_tokens = [tokenizer.cls_token]
    # bert_tokens = ["[CLS]"]

    bert_tokens.extend(tokens)
    label_ids = [pad_token_label_id] + label_ids

    bert_tokens.append(tokenizer.sep_token)
    # bert_tokens.append("[SEP]")
    label_ids += [pad_token_label_id]

    return bert_tokens, label_ids


def tokenize_with_new_mask(orig_text, max_length, tokenizer, orig_labels, label_map):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    pad_token_label_id = -100
    simple_tokenize_results = [list(tt) for tt in zip(
        *[simple_tokenize(orig_text[i], tokenizer, orig_labels[i], label_map, max_length) for i in
          range(len(orig_text))])]
    bert_tokens, label_ids = simple_tokenize_results[0], simple_tokenize_results[1]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attention_masks, label_ids


def train(model, optimizer, train_batch_generator, num_batches, device, args, label_map, class_weight):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, masks_batch = next(train_batch_generator)
        x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
        class_weight = class_weight.to(device) if class_weight is not None else None
        optimizer.zero_grad()

        outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch, class_weight=class_weight)

        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        if type(model) is RobertaForTokenClassificationWithCRF:
            y_batch = y_batch.detach().cpu()
            y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
            eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
        else:
            eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

        epoch_loss += loss.item()
        epoch_acc += eval_metrics["accuracy_score"]
        epoch_results.update(eval_metrics["results"])
        epoch_results_by_tag.update(eval_metrics["results_by_tag"])
        epoch_CR = eval_metrics["CR"]
    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, epoch_CR


def evaluate(model, test_batch_generator, num_batches, device, label_map, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    output_t_pred = None

    model.eval()
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            class_weight = class_weight.to(device) if class_weight is not None else None
            outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch,
                            class_weight=class_weight)

            loss, logits = outputs[:2]

            if type(model) is RobertaForTokenClassificationWithCRF:
                y_batch = y_batch.detach().cpu()
                y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
                eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
            else:
                eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

            epoch_loss += loss.item()
            epoch_acc += eval_metrics["accuracy_score"]
            epoch_results.update(eval_metrics["results"])
            epoch_results_by_tag.update(eval_metrics["results_by_tag"])
            epoch_CR = eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, logits.detach().cpu().numpy()], axis=0)

    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, output_t_pred, epoch_CR


def predict(model, test_batch_generator, num_batches, device, label_map, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_results = {}
    epoch_results_by_tag = {}
    epoch_CR = ""

    output_t_pred = None

    model.eval()
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            class_weight = class_weight.to(device) if class_weight is not None else None

            outputs = model(input_ids=x_batch, attention_mask=masks_batch, labels=y_batch,
                            class_weight=class_weight)

            loss, logits = outputs[:2]

            if type(model) is RobertaForTokenClassificationWithCRF:
                y_batch = y_batch.detach().cpu()
                y_batch_filtered = [y_batch[i][y_batch[i] >= 0].tolist() for i in range(y_batch.shape[0])]
                eval_metrics = compute_crf_metrics(outputs[2], y_batch_filtered, label_map)
            else:
                eval_metrics = compute_metrics(logits.detach().cpu(), y_batch.detach().cpu(), label_map)

            epoch_loss += loss.item()
            epoch_acc += eval_metrics["accuracy_score"]
            epoch_results.update(eval_metrics["results"])
            epoch_results_by_tag.update(eval_metrics["results_by_tag"])
            epoch_CR = eval_metrics["CR"]
            if output_t_pred is None:
                output_t_pred = logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred, logits.detach().cpu().numpy()], axis=0)

    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_results, \
           epoch_results_by_tag, output_t_pred, epoch_CR


def load_model(model_type, model_path, config):
    if model_type == 'bertweet-token':
        model = RobertaForWeightedTokenClassification.from_pretrained(model_path, config=config)
    elif model_type == 'bertweet-token-crf':
        model = RobertaForTokenClassificationWithCRF.from_pretrained(model_path, config=config)
    else:
        model = None
    return model



NOTE = 'V1.0.0: Initial Public Version'


### Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="vinai/bertweet-base", type=str)
    parser.add_argument("--model_type", default='bertweet-token', type=str)
    parser.add_argument("--task_type", default='entity_detection', type=str)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=300, type=int)
    parser.add_argument('--test_batch_size', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--data', default='wnut_16', type=str)
    parser.add_argument('--log_dir', default='log-BERTweet-token', type=str)
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=False, action='store_true')
    parser.add_argument("--assign_weight", default=False, action='store_true')
    parser.add_argument("--data_file", default=None, type=str)
    parser.add_argument("--label_map", default=None, type=str)
    parser.add_argument("--performance_file", default='all_test_performance.txt', type=str)

    args = parser.parse_args()

    assert args.model_type in ['bertweet-token', 'bertweet-token-crf',]
    assert args.task_type in ['entity_detection', 'relevant_entity_detection', 'entity_relevance_classification']

    print("cuda is available:", torch.cuda.is_available())
    log_directory = args.log_dir + '/' + args.bert_model.split('/')[-1] + '/' + args.model_type + '/' + \
                    args.task_type + '/' + str(args.n_epochs) + '_epoch/' + \
                    args.data.split('/')[-1] + '/' + str(args.assign_weight) + \
                    '_weight/' + str(args.seed) + '_seed/'
    log_filename = 'log.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-') + '.txt'
    per_filename = 'performance.csv'
    model_dir = 'saved-model'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logname = log_directory + log_filename
    modeldir = log_directory + model_dir
    perfilename = log_directory + per_filename

    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    if os.path.exists(modeldir) and os.listdir(modeldir):
        logging.info(f"modeldir {modeldir} already exists and it is not empty")
        print(f"modeldir {modeldir} already exists and it is not empty")
    else:
        os.makedirs(modeldir, exist_ok=True)
        logging.info(f"Create modeldir: {modeldir}")
        print(f"Create modeldir: {modeldir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    all_data = pd.read_pickle(os.path.join(args.data, args.data_file))
    train_index, val_test_index = train_test_split(all_data.index, test_size=0.2, random_state=args.seed)
    val_index, test_index = train_test_split(val_test_index, test_size=0.5, random_state=args.seed)
    need_columns = ['tweet_tokens']
    if args.task_type == 'entity_detection':

        need_columns.append('entity_label')
    elif args.task_type == 'relevant_entity_detection':
        need_columns.append('relevant_entity_label')
    elif args.task_type == 'entity_relevance_classification':
        need_columns.append('relevance_entity_class_label')
    need_columns.append('sentence_class')
    X_train_raw, Y_train_raw, seq_train = extract_from_dataframe(all_data, need_columns, train_index)
    X_dev_raw, Y_dev_raw, seq_dev = extract_from_dataframe(all_data, need_columns, val_index)
    X_test_raw, Y_test_raw, seq_test = extract_from_dataframe(all_data, need_columns, test_index)
    args.eval_batch_size = seq_dev.shape[0]
    args.test_batch_size = seq_test.shape[0]

    with open(os.path.join(args.data, args.label_map), 'r') as fp:
        label_map = json.load(fp)

    labels = list(label_map.keys())

    logging.info(args)
    print(args)


    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)

    X_train, masks_train, Y_train = tokenize_with_new_mask(
        X_train_raw, args.max_length, tokenizer, Y_train_raw, label_map)
    X_dev, masks_dev, Y_dev = tokenize_with_new_mask(
        X_dev_raw, args.max_length, tokenizer, Y_dev_raw, label_map)

    # weight of each class in loss function
    class_weight = None
    if args.assign_weight:
        class_weight = [Y_train.shape[0] / (Y_train == i).sum() for i in range(len(labels))]
        class_weight = torch.FloatTensor(class_weight)

    config = RobertaConfig.from_pretrained(args.bert_model)
    config.update({'num_labels': len(labels), })
    model = load_model(args.model_type, args.bert_model, config)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model = model.to(device)

    best_valid_acc = 0
    best_valid_P, best_valid_R, best_valid_F = 0, 0, 0
    train_losses = []
    eval_losses = []
    train_F_list, eval_F_list = [], []

    early_stop_sign = 0

    for epoch in range(args.n_epochs):

        # train

        train_batch_generator = mask_batch_generator(X_train, Y_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_loss, train_acc, train_results, train_results_by_tag, train_CR = train(model, optimizer,
                                                                                     train_batch_generator,
                                                                                     num_batches,
                                                                                     device, args, label_map,
                                                                                     class_weight)
        train_losses.append(train_loss)
        train_F = train_results['strict']['f1']
        train_P = train_results['strict']['precision']
        train_R = train_results['strict']['recall']
        train_F_list.append(train_F)

        # eval
        dev_batch_generator = mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                       min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        valid_loss, valid_acc, valid_results, valid_results_by_tag, valid_t_pred, valid_CR = evaluate(model,
                                                                                                      dev_batch_generator,
                                                                                                      num_batches,
                                                                                                      device,
                                                                                                      label_map,
                                                                                                      class_weight)
        eval_losses.append(valid_loss)
        valid_F = valid_results['strict']['f1']
        valid_P = valid_results['strict']['precision']
        valid_R = valid_results['strict']['recall']
        eval_F_list.append(valid_F)

        if best_valid_F < valid_F or epoch == 0:
            best_valid_acc = valid_acc
            best_valid_P = valid_P
            best_valid_R = valid_R
            best_valid_F = valid_F
            best_valid_results = valid_results
            best_valid_results_by_tag = valid_results_by_tag
            best_valid_CR = valid_CR

            best_train_acc = train_acc
            best_train_P = train_P
            best_train_R = train_R
            best_train_F = train_F
            best_train_results = train_results
            best_train_results_by_tag = train_results_by_tag
            best_train_CR = train_CR

            model.save_pretrained(modeldir)
            if args.early_stop:
                early_stop_sign = 0
        elif args.early_stop:
            early_stop_sign += 1

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Train P: {train_P * 100:.2f}%')
        print(f'Train R: {train_R * 100:.2f}%')
        print(f'Train F1: {train_F * 100:.2f}%')
        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        print(f'Val. P: {valid_P * 100:.2f}%')
        print(f'Val. R: {valid_R * 100:.2f}%')
        print(f'Val. F1: {valid_F * 100:.2f}%')
        logging.info(f'Train. Acc: {train_acc * 100:.2f}%')
        logging.info(f'Train. P: {train_P * 100:.2f}%')
        logging.info(f'Train. R: {train_R * 100:.2f}%')
        logging.info(f'Train. F1: {train_F * 100:.2f}%')
        logging.info(f'Val. Acc: {valid_acc * 100:.2f}%')
        logging.info(f'Val. P: {valid_P * 100:.2f}%')
        logging.info(f'Val. R: {valid_R * 100:.2f}%')
        logging.info(f'Val. F1: {valid_F * 100:.2f}%')

        if args.early_stop and early_stop_sign >= 5:
            break

    content = f"After {epoch + 1} epoch, Best valid F1: {best_valid_F}, accuracy: {best_valid_acc}, Recall: {best_valid_R}, Precision: {best_valid_P}"
    print(content)
    logging.info(content)

    performance_dict = vars(args)
    performance_dict['T_best_train_F'] = best_train_F
    performance_dict['T_best_train_ACC'] = best_train_acc
    performance_dict['T_best_train_R'] = best_train_R
    performance_dict['T_best_train_P'] = best_train_P
    performance_dict['T_best_train_CR'] = best_train_CR
    performance_dict['T_best_train_results'] = best_train_results
    performance_dict['T_best_train_results_by_tag'] = best_train_results_by_tag

    performance_dict['T_best_valid_F'] = best_valid_F
    performance_dict['T_best_valid_ACC'] = best_valid_acc
    performance_dict['T_best_valid_R'] = best_valid_R
    performance_dict['T_best_valid_P'] = best_valid_P
    performance_dict['T_best_valid_CR'] = best_valid_CR
    performance_dict['T_best_valid_results'] = best_valid_results
    performance_dict['T_best_valid_results_by_tag'] = best_valid_results_by_tag

    # Plot training classification loss
    epoch_count = np.arange(1, epoch + 2)
    fig, axs = plt.subplots(2, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0].plot(epoch_count, train_losses, 'b--')
    axs[0].plot(epoch_count, eval_losses, 'b-')
    axs[0].legend(['Training Loss', 'Valid Loss'], fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14, labelcolor='b')
    axs[0].tick_params(axis='x', labelsize=14)
    axs[1].plot(epoch_count, train_F_list, 'r--')
    axs[1].plot(epoch_count, eval_F_list, 'r-')
    axs[1].legend(['Training F1', 'Valid F1'], fontsize=14)
    axs[1].set_ylabel('F1', fontsize=16)
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14, labelcolor='r')
    axs[1].tick_params(axis='x', labelsize=14)

    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)
    X_test, masks_test, Y_test = tokenize_with_new_mask(
        X_test_raw, 128, tokenizer, Y_test_raw, label_map)
    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)
    del model
    torch.cuda.empty_cache()
    model = load_model(args.model_type, modeldir, config)
    model = model.to(device)
    test_loss, test_acc, test_results, test_results_by_tag, test_t_pred, test_CR = predict(model,
                                                                                           test_batch_generator,
                                                                                           num_batches, device,
                                                                                           label_map, class_weight)
    test_F = test_results['strict']['f1']
    test_P = test_results['strict']['precision']
    test_R = test_results['strict']['recall']
    print(f'Test Acc: {test_acc * 100:.2f}%')
    print(f'Test P: {test_P * 100:.2f}%')
    print(f'Test R: {test_R * 100:.2f}%')
    print(f'Test F1: {test_F * 100:.2f}%')
    logging.info(f'Test Acc: {test_acc * 100:.2f}%')
    logging.info(f'Test P: {test_P * 100:.2f}%')
    logging.info(f'Test R: {test_R * 100:.2f}%')
    logging.info(f'Test F1: {test_F * 100:.2f}%')
    token_pred_dir = log_directory + 'token_prediction.npy'
    np.save(token_pred_dir, test_t_pred)

    performance_dict['T_best_test_F'] = test_F
    performance_dict['T_best_test_ACC'] = test_acc
    performance_dict['T_best_test_R'] = test_R
    performance_dict['T_best_test_P'] = test_P
    performance_dict['T_best_test_CR'] = test_CR
    performance_dict['T_best_test_results'] = test_results
    performance_dict['T_best_test_results_by_tag'] = test_results_by_tag

    performance_dict['script_file'] = os.path.basename(__file__)
    performance_dict['log_directory'] = log_directory
    performance_dict['log_filename'] = log_filename
    performance_dict['note'] = NOTE
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['device'] = torch.cuda.get_device_name(device)
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
    with open(args.performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')
    if not args.save_model:
        shutil.rmtree(modeldir)


if __name__ == '__main__':
    main()
