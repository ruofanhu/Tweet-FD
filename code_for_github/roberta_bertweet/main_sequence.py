#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import random
import argparse
import datetime
import logging
import os, os.path
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from transformers import AdamW, AutoTokenizer, RobertaConfig
from model_weighted_roberta import *
import json


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    values, indices = torch.max(probabilities, 1)
    y_pred = indices
    acc = accuracy_score(y, y_pred)
    return acc


def eval_metrics(preds, y):
    '''
    Returns performance metrics of predictor
    :param y: ground truth label
    :param preds: predicted logits
    :return: auc, acc, tn, fp, fn, tp
    '''
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    y_values, indices = torch.max(probabilities, 1)
    y_pred = indices
    try:
        auc = roc_auc_score(y, y_values)
    except:
        auc = np.array(0)
    acc = accuracy_score(y, y_pred)
    conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    return auc, acc, tn, fp, fn, tp


def simple_tokenize(orig_tokens, tokenizer):
    """
    tokenize a array of raw text
    """
    # bert_tokens = ["[CLS]"]
    bert_tokens = [tokenizer.cls_token]
    for x in orig_tokens:
        bert_tokens.extend(tokenizer.tokenize(x))
    # bert_tokens.append("[SEP]")
    bert_tokens.append(tokenizer.sep_token)
    return bert_tokens


def tokenize_with_new_mask(orig_text, max_length, tokenizer):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    bert_tokens = [simple_tokenize(t, tokenizer) for t in orig_text]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attention_masks


def train(model, optimizer, train_batch_generator, num_batches, device, class_weight):
    """
    Main training routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    epoch_cl = 0
    epoch_al = 0
    epoch_tn, epoch_fp, epoch_fn, epoch_tp = 0, 0, 0, 0

    model.train()

    # Training
    for b in tqdm(range(num_batches)):
        x_batch, y_batch, masks_batch = next(train_batch_generator)
        if len(x_batch.shape) == 3:
            x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
        else:
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
        y_batch = y_batch.astype(np.float)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
        class_weight = class_weight.to(device) if class_weight is not None else None
        optimizer.zero_grad()
        outputs = model(x_batch, masks_batch, labels=y_batch, class_weight=class_weight)

        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        auc, acc, tn, fp, fn, tp = eval_metrics(logits.detach().cpu(),
                                                y_batch.detach().cpu())

        epoch_loss += loss.item()
        epoch_auc += auc.item()
        epoch_acc += acc.item()
        epoch_tn += tn.item()
        epoch_fp += fp.item()
        epoch_fn += fn.item()
        epoch_tp += tp.item()
    print(f'\tClassification Loss: {epoch_loss / num_batches:.3f}')
    return epoch_loss / num_batches, epoch_auc / num_batches, epoch_acc / num_batches, epoch_tn, epoch_fp, epoch_fn, epoch_tp


def evaluate(model, test_batch_generator, num_batches, device, class_weight):
    """
    Main evaluation routine
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    epoch_tn, epoch_fp, epoch_fn, epoch_tp = 0, 0, 0, 0

    output_s_pred = None

    model.eval()

    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, y_batch, masks_batch = next(test_batch_generator)
            if len(x_batch.shape) == 3:
                x_batch = Variable(torch.FloatTensor(x_batch)).to(device)
            else:
                x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            y_batch = y_batch.astype(np.float)
            y_batch = Variable(torch.LongTensor(y_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            class_weight = class_weight.to(device) if class_weight is not None else None
            outputs = model(x_batch, masks_batch, labels=y_batch, class_weight=class_weight)

            loss, logits = outputs[:2]

            auc, acc, tn, fp, fn, tp = eval_metrics(logits.detach().cpu(),
                                                    y_batch.detach().cpu())

            epoch_loss += loss.item()
            epoch_auc += auc.item()
            epoch_acc += acc.item()
            epoch_tn += tn.item()
            epoch_fp += fp.item()
            epoch_fn += fn.item()
            epoch_tp += tp.item()

            if output_s_pred is None:
                output_s_pred = logits.detach().cpu().numpy()
            else:
                output_s_pred = np.concatenate([output_s_pred, logits.detach().cpu().numpy()], axis=0)
    return epoch_loss / num_batches, epoch_auc / num_batches, epoch_acc / num_batches, epoch_tn, epoch_fp, epoch_fn, epoch_tp, output_s_pred


def single_evaluate(model, x_item, y_item, mask_item, device):
    """
    Evaluates a single instance and ALSO returns the attention scores for this instance.
    Use this routine for attention evaluation
    """
    model.eval()

    with torch.no_grad():
        x_item = Variable(torch.LongTensor(x_item)).to(device)
        y_item = y_item.astype(np.float)
        y_item = Variable(torch.LongTensor(y_item)).to(device)
        mask_item = Variable(torch.FloatTensor(mask_item)).to(device)

        outputs = model(x_item, mask_item, labels=y_batch)

        loss, logits = outputs[:2]

        auc, acc, tn, fp, fn, tp = eval_metrics(logits.detach().cpu(),
                                                y_item.detach().cpu())
        epoch_acc = acc.item()
        epoch_auc = auc.item()
        epoch_tn = tn.item()
        epoch_fp = fp.item()
        epoch_fn = fn.item()
        epoch_tp = tp.item()

    return epoch_auc, epoch_acc, epoch_tn, epoch_fp, epoch_fn, epoch_tp


def load_model(model_type, model_path, config):
    if model_type == 'bertweet-seq':
        model = RobertaForWeightedSequenceClassification.from_pretrained(model_path, config=config)
    elif model_type == 'BiLSTM-seq':
        model = BiLSTMForWeightedSequenceClassification(config=config)
        if model_path is not None:
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.pt')))
    else:
        model = None
    return model


def get_partial_labeled_data(X_train, Y_train, masks_train, args):
    partial_percent = args.seq_percent
    if partial_percent == 1.0:
        X_train_multi, Y_train_multi, \
        masks_train_multi = X_train, Y_train, masks_train
        X_train_single, Y_train_single, \
        masks_train_single = None, None, None
    else:
        if args.pos_missing == True:
            rel_index = pd.Series(Y_train).loc[Y_train == 1].index
            all_index = pd.Series(Y_train).index
            single_index, _ = train_test_split(rel_index,
                                               test_size=partial_percent,
                                               random_state=args.seed)
            multi_index = all_index[~all_index.isin(single_index)]

            X_train_single = X_train[single_index]
            Y_train_single = Y_train[single_index]
            masks_train_single = masks_train[single_index]

            X_train_multi = X_train[multi_index]
            Y_train_multi = Y_train[multi_index]
            masks_train_multi = masks_train[multi_index]
        else:
            X_train_single, X_train_multi, Y_train_single, Y_train_multi, \
            masks_train_single, masks_train_multi = train_test_split(X_train,
                                                                     Y_train,
                                                                     masks_train,
                                                                     test_size=partial_percent,
                                                                     random_state=args.seed)
    return X_train_single, Y_train_single, masks_train_single, \
           X_train_multi, Y_train_multi, masks_train_multi,


def get_embedding(text_list, embeddings_index, embeddings, max_length):
    output_embedding = []
    attention_masks_list = []
    for words in text_list:
        words_mapped = [0] * max_length
        attention_mask = [1] * max_length
        length = len(words)
        if (length < max_length):
            for i in range(0, length):
                words_mapped[i] = embeddings_index.get(words[i], -1)
            for i in range(length, max_length):
                words_mapped[i] = -2
                attention_mask[i] = 0
        elif (length > max_words):
            print('We should never see this print either')
        else:
            for i in range(0, max_length):
                words_mapped[i] = embeddings_index.get(words[i], -1)

        output_embedding.append(np.array([embeddings[ix] for ix in words_mapped]))
        attention_masks_list.append(attention_mask)
    output_embedding = np.array(output_embedding)
    attention_masks_list = np.array(attention_masks_list)
    return output_embedding, attention_masks_list


NOTE = 'V1.0.0: Initial Public Version'


### Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=None, type=str)
    parser.add_argument("--model_type", default=None, type=str)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--rnn_hidden_size', default=384, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=300, type=int)
    parser.add_argument('--test_batch_size', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--data', default='wnut_16', type=str)
    parser.add_argument('--log_dir', default='log-BERTweet-seq', type=str)
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--early_stop", default=False, action='store_true')
    parser.add_argument("--assign_weight", default=False, action='store_true')
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--val_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--performance_file", default='all_test_performance.txt', type=str)
    parser.add_argument("--embeddings_file", default='glove.840B.300d.txt', type=str)

    args = parser.parse_args()

    print("cuda is available:", torch.cuda.is_available())
    log_directory = args.log_dir + '/' + str(args.bert_model).split('/')[-1] + '/' + args.model_type + '/' \
                    + str(args.n_epochs) + '_epoch/' + args.data.split('/')[-1] + '/' + \
                    str(args.assign_weight) + '_weight/' + str(args.seed) + '_seed/'
    log_filename = 'log.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.', '-') + '.txt'
    per_filename = 'performance.csv'
    model_dir = 'saved-model'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logname = log_directory + log_filename
    modeldir = log_directory + model_dir

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


    train_data = pd.read_pickle(os.path.join(args.data, args.train_file))
    val_data = pd.read_pickle(os.path.join(args.data, args.val_file))
    test_data = pd.read_pickle(os.path.join(args.data, args.test_file))
    need_columns = ['tweet_tokens', 'sentence_class']
    X_train_raw, Y_train = extract_from_dataframe(train_data, need_columns)
    X_dev_raw, Y_dev = extract_from_dataframe(val_data, need_columns)
    X_test_raw, Y_test = extract_from_dataframe(test_data, need_columns)
    args.eval_batch_size = Y_dev.shape[0]
    args.test_batch_size = Y_test.shape[0]


    logging.info(args)
    print(args)

    if args.bert_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, normalization=True)
        X_train, masks_train = tokenize_with_new_mask(
            X_train_raw, args.max_length, tokenizer)
        X_dev, masks_dev = tokenize_with_new_mask(
            X_dev_raw, args.max_length, tokenizer)
        X_test, masks_test = tokenize_with_new_mask(
            X_test_raw, args.max_length, tokenizer)
    else:
        embeddings_index, embeddings = new_build_glove_embedding(
            embedding_path=args.embeddings_file)
        X_train, masks_train = get_embedding(X_train_raw, embeddings_index, embeddings, args.max_length)
        X_dev, masks_dev = get_embedding(X_dev_raw, embeddings_index, embeddings, args.max_length)
        X_test, masks_test = get_embedding(X_test_raw, embeddings_index, embeddings, args.max_length)

    # weight of each class in loss function
    class_weight = None
    if args.assign_weight:
        class_weight = [np.array(Y_train).shape[0] / (np.array(Y_train) == i).sum() for i in range(len(set(Y_train)))]
        class_weight = torch.FloatTensor(class_weight)

    if args.bert_model is not None:
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.update({'num_labels': len(set(Y_train))})
    else:
        from dotmap import DotMap
        config = DotMap()
        config.update({'num_labels': len(set(Y_train)),
                       'rnn_hidden_dimension': args.rnn_hidden_size,
                       'hidden_dropout_prob': 0.1})
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
    best_valid_auc = 0
    best_valid_tn, best_valid_fp, best_valid_fn, best_valid_tp = 0, 0, 0, 0
    train_losses = []
    eval_losses = []
    train_acc_list, eval_acc_list = [], []

    early_stop_sign = 0

    for epoch in range(args.n_epochs):

        # train
        train_batch_generator = mask_batch_generator(X_train, Y_train, masks_train, args.batch_size)
        num_batches = X_train.shape[0] // args.batch_size
        train_loss, train_auc, train_acc, train_tn, train_fp, train_fn, train_tp = train(model, optimizer,
                                                                                         train_batch_generator,
                                                                                         num_batches, device,
                                                                                         class_weight)
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        # eval
        dev_batch_generator = mask_batch_seq_generator(X_dev, Y_dev, masks_dev,
                                                       min(X_dev.shape[0], args.eval_batch_size))
        num_batches = X_dev.shape[0] // min(X_dev.shape[0], args.eval_batch_size)
        valid_loss, valid_auc, valid_acc, valid_tn, valid_fp, valid_fn, valid_tp, valid_s_pred = evaluate(model,
                                                                                                          dev_batch_generator,
                                                                                                          num_batches,
                                                                                                          device,
                                                                                                          class_weight)
        eval_losses.append(valid_loss)
        eval_acc_list.append(valid_acc)

        if best_valid_acc < valid_acc or epoch == 0:
            best_valid_acc = valid_acc
            best_valid_auc = valid_auc
            best_valid_tn = valid_tn
            best_valid_fp = valid_fp
            best_valid_fn = valid_fn
            best_valid_tp = valid_tp

            best_train_auc = train_auc
            best_train_acc = train_acc
            best_train_tn = train_tn
            best_train_fp = train_fp
            best_train_fn = train_fn
            best_train_tp = train_tp

            if args.bert_model is not None:
                model.save_pretrained(modeldir)
            else:
                torch.save(model.state_dict(), os.path.join(modeldir, 'pytorch_model.pt'))
            if args.early_stop:
                early_stop_sign = 0
        elif args.early_stop:
            early_stop_sign += 1

        print(f'Train Acc: {train_acc * 100:.2f}%')
        print(f'Val. Acc: {valid_acc * 100:.2f}%')
        logging.info(f'Train. Acc: {train_acc * 100:.2f}%')
        logging.info(f'Val. Acc: {valid_acc * 100:.2f}%')
        if args.early_stop and early_stop_sign >= 5:
            break

    print(f"After {epoch + 1} epoch, Best valid accuracy: {best_valid_acc}")
    logging.info(f"After {epoch + 1} epoch, Best valid accuracy: {best_valid_acc}")

    performance_dict = vars(args)
    performance_dict['S_best_train_AUC'] = best_train_auc
    performance_dict['S_best_train_ACC'] = best_train_acc
    performance_dict['S_best_train_TN'] = best_train_tn
    performance_dict['S_best_train_FP'] = best_train_fp
    performance_dict['S_best_train_FN'] = best_train_fn
    performance_dict['S_best_train_TP'] = best_train_tp

    performance_dict['S_best_valid_AUC'] = best_valid_auc
    performance_dict['S_best_valid_ACC'] = best_valid_acc
    performance_dict['S_best_valid_TN'] = best_valid_tn
    performance_dict['S_best_valid_FP'] = best_valid_fp
    performance_dict['S_best_valid_FN'] = best_valid_fn
    performance_dict['S_best_valid_TP'] = best_valid_tp

    # Plot training classification loss
    epoch_count = np.arange(1, epoch + 2)
    fig, axs = plt.subplots(2, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0].plot(epoch_count, train_losses, 'b--')
    axs[0].plot(epoch_count, eval_losses, 'b-')
    axs[0].legend(['Training Loss', 'Valid Loss'], fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14, labelcolor='b')
    axs[0].tick_params(axis='x', labelsize=14)

    axs[1].plot(epoch_count, train_acc_list, 'y--')
    axs[1].plot(epoch_count, eval_acc_list, 'y-')
    axs[1].legend(['Training Acc', 'Valid Acc'], fontsize=14)
    axs[1].set_ylabel('Acc', fontsize=16)
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14, labelcolor='y')
    axs[1].tick_params(axis='x', labelsize=14)

    figure_filename = 'fig.' + str(datetime.datetime.now()).replace(' ', '--').replace(':', '-').replace('.',
                                                                                                         '-') + '.png'
    figfullname = log_directory + figure_filename
    plt.savefig(figfullname, dpi=fig.dpi)


    num_batches = X_test.shape[0] // args.test_batch_size
    test_batch_generator = mask_batch_seq_generator(X_test, Y_test, masks_test, args.test_batch_size)

    del model
    torch.cuda.empty_cache()
    model = load_model(args.model_type, modeldir, config)
    model = model.to(device)
    test_loss, test_auc, test_acc, test_tn, test_fp, test_fn, test_tp, test_s_pred = evaluate(model,
                                                                                              test_batch_generator,
                                                                                              num_batches, device,
                                                                                              class_weight)

    content = f'Test Acc: {test_acc * 100:.2f}%, AUC: {test_auc * 100:.2f}%, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}'
    print(content)
    logging.info(content)
    seq_pred_dir = log_directory + 'seq_prediction.npy'
    np.save(seq_pred_dir, test_s_pred)

    performance_dict['S_best_test_AUC'] = test_auc
    performance_dict['S_best_test_ACC'] = test_acc
    performance_dict['S_best_test_TN'] = test_tn
    performance_dict['S_best_test_FP'] = test_fp
    performance_dict['S_best_test_FN'] = test_fn
    performance_dict['S_best_test_TP'] = test_tp

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
