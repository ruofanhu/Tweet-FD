'''
Training script for IMGJM
'''
from typing import Dict, Tuple
from argparse import ArgumentParser
import logging
import yaml
import coloredlogs
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from IMGJM import IMGJM
from IMGJM.new_data import Twitter
from IMGJM.utils import new_build_glove_embedding
import seqeval.metrics
from nervaluate import Evaluator
import json
import datetime
import os

NOTE = 'V1.0.0: initial version for IMGJM code'


class BoolParser:
    @classmethod
    def parse(cls, arg: str) -> bool:
        if arg.lower() in ['false', 'no']:
            return False
        else:
            return True


def get_logger(logger_name: str = 'IMGJM',
               level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    coloredlogs.install(
        level=level,
        fmt=
        f'%(asctime)s | %(name)-{len(logger_name) + 1}s| %(levelname)s | %(message)s',
        logger=logger)
    return logger


def get_args() -> Dict:
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--valid_batch_size', type=int, default=300)
    arg_parser.add_argument('--test_batch_size', type=int, default=300)
    arg_parser.add_argument('--epochs', type=int, default=20)
    arg_parser.add_argument('--seed', type=int, default=42)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--model_config_fp',
                            type=str,
                            default='new_model_settings.yml')
    arg_parser.add_argument('--embedding', type=str, default='glove')
    arg_parser.add_argument('--embedding_path', type=str, default='glove')
    arg_parser.add_argument('--dataset', type=str, default='Twitter')
    arg_parser.add_argument("--performance_file", default='all_test_performance.txt', type=str)
    arg_parser.add_argument("--early_stop", default=False, action='store_true')
    return vars(arg_parser.parse_args())


def build_feed_dict(input_tuple: Tuple[np.ndarray],
                    input_type: str = 'ids') -> Dict:
    if input_type == 'ids':
        pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities = input_tuple
        feed_dict = {
            'char_ids': pad_char_ids,
            'word_ids': pad_word_ids,
            'sequence_length': sequence_length,
            'y_target': pad_entities,
            'y_sentiment': pad_polarities,
        }
        return feed_dict
    else:
        pad_char_ids, pad_word_embedding, sequence_length, pad_entities, pad_polarities = input_tuple
        feed_dict = {
            'char_ids': pad_char_ids,
            'word_embedding': pad_word_embedding,
            'sequence_length': sequence_length,
            'y_target': pad_entities,
            'y_sentiment': pad_polarities,
        }
        return feed_dict


def load_model_config(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def to_text_label(x, label_map_switch):
    return label_map_switch[x]


v_to_text_label = np.vectorize(to_text_label)


def compute_metrics(preds: np.ndarray,
                    label_ids: np.ndarray,
                    label_map: Dict,
                    label_map_switch: Dict) -> Dict:
    text_preds = v_to_text_label(preds, label_map_switch)
    text_label_ids = v_to_text_label(label_ids, label_map_switch)
    text_preds = text_preds.tolist()
    text_label_ids = text_label_ids.tolist()
    labels = list(label_map.keys())
    labels = [i[2:] for i in labels if i.startswith('B-')]
    evaluator = Evaluator(text_label_ids, text_preds, tags=labels, loader="list")
    results, results_by_tag = evaluator.evaluate()
    try:
        cls_report = seqeval.metrics.classification_report(text_label_ids, text_preds, zero_division=1)
    except:
        cls_report = ""
    return {
        "acc": seqeval.metrics.accuracy_score(text_label_ids, text_preds),
        "results": results,
        "results_by_tag": results_by_tag,
        "CR": cls_report,
    }


def get_computed_metrics(feed_dict: Dict,
                         dataset,
                         model: IMGJM,
                         C_tar: int = 5,
                         C_sent: int = 7,
                         *args,
                         **kwargs) -> Tuple[dict, dict]:
    target_preds, sentiment_preds = model.predict_on_batch(feed_dict)
    target_labels = feed_dict.get('y_target')
    sentiment_labels = feed_dict.get('y_sentiment')

    label_map_switch = {dataset.label_map[k]: k for k in dataset.label_map}
    re_label_map_switch = {dataset.re_label_map[k]: k for k in dataset.re_label_map}
    target_metrics = compute_metrics(target_preds, target_labels, dataset.label_map, label_map_switch)
    sentiment_metrics = compute_metrics(sentiment_preds, sentiment_labels, dataset.re_label_map, re_label_map_switch)
    return target_metrics, sentiment_metrics


def get_confusion_matrix(feed_dict: Dict,
                         model: IMGJM,
                         C_tar: int = 5,
                         C_sent: int = 7,
                         *args,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get target and sentiment confusion matrix

    Args:
        feed_dict (dict): model inputs.
        model (IMGJM): model.
        C_tar (int): target class numbers.
        C_sent (int): sentiment class numbers.

    Returns:
        target_cm (np.ndarray): target confusion matrix.
        sentiment_cm (np.ndarray): sentiment confusion matrix.
    '''
    target_preds, sentiment_preds = model.predict_on_batch(feed_dict)
    target_labels = feed_dict.get('y_target')
    sentiment_labels = feed_dict.get('y_sentiment')

    target_confusion_matrix = confusion_matrix(
        np.reshape(target_labels,
                   (target_labels.shape[0] * target_labels.shape[1])),
        np.reshape(target_preds,
                   (target_preds.shape[0] * target_preds.shape[1])),
        labels=list(range(C_tar)))
    sentiment_confusion_matrix = confusion_matrix(
        np.reshape(sentiment_labels,
                   (sentiment_labels.shape[0] * sentiment_labels.shape[1])),
        np.reshape(sentiment_preds,
                   (sentiment_preds.shape[0] * sentiment_preds.shape[1])),
        labels=list(range(C_sent)))
    return target_confusion_matrix, sentiment_confusion_matrix


def main(*args, **kwargs):
    performance_dict = kwargs.copy()
    print(performance_dict)
    np.random.seed(kwargs.get('seed'))
    logger = get_logger()
    if kwargs.get('embedding') == 'glove':
        logger.info('Loading Glove embedding...')
        word2id, embedding_weights, _ = new_build_glove_embedding(embedding_path=kwargs.get('embedding_path'))
        logger.info('Embeding loaded.')
        logger.info('Initializing dataset...')
        dataset = Twitter(word2id=word2id, seed=kwargs['seed'])
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    else:
        logger.warning('Invalid embedding choice.')
    logger.info('Loading model...')
    config = load_model_config(kwargs.get('model_config_fp'))
    performance_dict.update(config)
    if kwargs.get('embedding') == 'fasttext':
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_size=dataset.fasttext_model.get_dimension(),
                      input_type='embedding',
                      dropout=False,
                      **config['custom'])
    else:
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_weights=embedding_weights,
                      dropout=False,
                      **config['custom'])
    logger.info('Model loaded.')
    logger.info('Start training...')
    C_tar = config['custom'].get('C_tar')
    C_sent = config['custom'].get('C_sent')

    best_valid_tar_p, best_valid_tar_r, best_valid_tar_f1, best_valid_sent_p, best_valid_sent_r, best_valid_sent_f1 = 0, 0, 0, 0, 0, 0
    best_valid_tar_acc, best_valid_sent_acc = 0, 0
    early_stop_sign = 0
    for epoch in trange(kwargs.get('epochs'), desc='epoch'):
        # Train
        train_batch_generator = tqdm(
            dataset.batch_generator(batch_size=kwargs.get('batch_size'), data_source='train'),
            desc='training')
        target_cm, sentiment_cm = np.zeros(
            (1, C_tar, C_tar), dtype=np.int32), np.zeros((1, C_sent, C_sent),
                                                         dtype=np.int32)
        for input_tuple in train_batch_generator:
            if kwargs.get('embedding') == 'fasttext':
                feed_dict = build_feed_dict(input_tuple,
                                            input_type='embedding')
            else:
                feed_dict = build_feed_dict(input_tuple, input_type='ids')
            tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.train_on_batch(
                feed_dict)
            temp_target_cm, temp_sentiment_cm = get_confusion_matrix(
                feed_dict, model, C_tar=C_tar, C_sent=C_sent)
            target_metrics, sentiment_metrics = get_computed_metrics(
                feed_dict, dataset, model, C_tar=C_tar, C_sent=C_sent)

            tar_acc, tar_p, tar_r, tar_f1, tar_cr = target_metrics['acc'], \
                                                    target_metrics['results']['strict']['precision'], \
                                                    target_metrics['results']['strict']['recall'], \
                                                    target_metrics['results']['strict']['f1'], \
                                                    target_metrics['CR']

            sent_acc, sent_p, sent_r, sent_f1, sent_cr = sentiment_metrics['acc'], \
                                                         sentiment_metrics['results']['strict']['precision'], \
                                                         sentiment_metrics['results']['strict']['recall'], \
                                                         sentiment_metrics['results']['strict']['f1'], \
                                                         sentiment_metrics['CR']
            target_cm = np.append(target_cm,
                                  np.expand_dims(temp_target_cm, axis=0),
                                  axis=0)
            sentiment_cm = np.append(sentiment_cm,
                                     np.expand_dims(temp_sentiment_cm, axis=0),
                                     axis=0)
            train_batch_generator.set_description(
                f'[Train][Entity]: a-{tar_acc:.3f}, p-{tar_p:.3f}, r-{tar_r:.3f}, f1-{tar_f1:.3f} [Relevance]: a-{sent_acc:.3f}, p-{sent_p:.3f}, r-{sent_r:.3f}, f1-{sent_f1:.3f}'
            )
        train_batch_generator.write(
            f'[Train][Entity][Report]:\n {tar_cr}')
        train_batch_generator.write(
            f'[Train][Relevance][Report]:\n {sent_cr}')
        train_batch_generator.write(
            f'[Train][Entity][CM]:\n {str(np.sum(target_cm, axis=0))}')
        train_batch_generator.write(
            f'[Train][Relevance][CM]:\n {str(np.sum(sentiment_cm, axis=0))}')

        # valid
        valid_batch_generator = tqdm(dataset.batch_generator(
            batch_size=kwargs.get('valid_batch_size'), data_source='dev'),
            desc='validation')
        target_cm, sentiment_cm = np.zeros(
            (1, C_tar, C_tar), dtype=np.int32), np.zeros((1, C_sent, C_sent),
                                                         dtype=np.int32)
        for input_tuple in valid_batch_generator:
            if kwargs.get('embedding') == 'fasttext':
                feed_dict = build_feed_dict(input_tuple,
                                            input_type='embedding')
            else:
                feed_dict = build_feed_dict(input_tuple, input_type='ids')
            tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.test_on_batch(
                feed_dict)
            temp_target_cm, temp_sentiment_cm = get_confusion_matrix(
                feed_dict, model, C_tar=C_tar, C_sent=C_sent)
            target_metrics, sentiment_metrics = get_computed_metrics(
                feed_dict, dataset, model, C_tar=C_tar, C_sent=C_sent)

            tar_acc, tar_p, tar_r, tar_f1, tar_cr = target_metrics['acc'], \
                                                    target_metrics['results']['strict']['precision'], \
                                                    target_metrics['results']['strict']['recall'], \
                                                    target_metrics['results']['strict']['f1'], \
                                                    target_metrics['CR']

            sent_acc, sent_p, sent_r, sent_f1, sent_cr = sentiment_metrics['acc'], \
                                                         sentiment_metrics['results']['strict']['precision'], \
                                                         sentiment_metrics['results']['strict']['recall'], \
                                                         sentiment_metrics['results']['strict']['f1'], \
                                                         sentiment_metrics['CR']

            tar_results, tar_results_by_tag = target_metrics['results'], target_metrics['results_by_tag']
            sent_results, sent_results_by_tag = sentiment_metrics['results'], sentiment_metrics['results_by_tag']

            target_cm = np.append(target_cm,
                                  np.expand_dims(temp_target_cm, axis=0),
                                  axis=0)
            sentiment_cm = np.append(sentiment_cm,
                                     np.expand_dims(temp_sentiment_cm, axis=0),
                                     axis=0)
            valid_batch_generator.set_description(
                f'[Valid][Entity]: a-{tar_acc:.3f}, p-{tar_p:.3f}, r-{tar_r:.3f}, f1-{tar_f1:.3f} [Relevance]: a-{sent_acc:.3f}, p-{sent_p:.3f}, r-{sent_r:.3f}, f1-{sent_f1:.3f}'
            )
            valid_batch_generator.write(
                f'[Valid][Entity][Report]:\n {tar_cr}')
            valid_batch_generator.write(
                f'[Valid][Relevance][Report]:\n {sent_cr}')
            valid_batch_generator.write(
                f'[Valid][Entity][CM]:\n {str(np.sum(target_cm, axis=0))}')
            valid_batch_generator.write(
                f'[Valid][Relevance][CM]:\n {str(np.sum(sentiment_cm, axis=0))}')
            good_cond_tar = best_valid_tar_f1 < tar_f1
            good_cond_sent = best_valid_sent_f1 < sent_f1
            normal_cond_tar = good_cond_tar or (abs(best_valid_tar_f1 - tar_f1) < 0.03)
            normal_cond_sent = good_cond_sent or (abs(best_valid_sent_f1 - sent_f1) < 0.03)
            if (epoch == 0) or (good_cond_tar and normal_cond_sent) or (good_cond_sent and normal_cond_tar):
                best_valid_tar_acc, best_valid_sent_acc = tar_acc, sent_acc
                best_valid_tar_p, best_valid_tar_r, best_valid_tar_f1 = tar_p, tar_r, tar_f1
                best_valid_sent_p, best_valid_sent_r, best_valid_sent_f1 = sent_p, sent_r, sent_f1
                best_valid_target_cm, best_valid_sentiment_cm = target_cm, sentiment_cm
                best_valid_tar_results, best_valid_tar_results_by_tag = tar_results, tar_results_by_tag
                best_valid_sent_results, best_valid_sent_results_by_tag = sent_results, sent_results_by_tag
                best_valid_tar_cr, best_valid_sent_cr = tar_cr, sent_cr
                model.save_model(kwargs.get('model_dir') + '/' + 'model')
            else:
                if kwargs['early_stop']:
                    early_stop_sign += 1
            if kwargs['early_stop'] and early_stop_sign >= 5:
                break

    performance_dict['best_valid_tar_acc'], performance_dict['best_valid_tar_p'], performance_dict['best_valid_tar_r'], \
    performance_dict['best_valid_tar_f1'], performance_dict['best_valid_tar_CR'], \
    performance_dict['best_valid_tar_results'], performance_dict['best_valid_tar_results_by_tag'] \
        = best_valid_tar_acc, best_valid_tar_p, best_valid_tar_r, best_valid_tar_f1, best_valid_tar_cr, \
          best_valid_tar_results, best_valid_tar_results_by_tag

    performance_dict['best_valid_sent_acc'], performance_dict['best_valid_sent_p'], \
    performance_dict['best_valid_sent_r'], performance_dict['best_valid_sent_f1'], performance_dict[
        'best_valid_sent_CR'], \
    performance_dict['best_valid_sent_results'], performance_dict['best_valid_sent_results_by_tag'] \
        = best_valid_sent_acc, best_valid_sent_p, best_valid_sent_r, best_valid_sent_f1, best_valid_sent_cr, \
          best_valid_sent_results, best_valid_sent_results_by_tag

    # Test
    model.load_model(kwargs.get('model_dir') + '/' + 'model')
    test_batch_generator = tqdm(dataset.batch_generator(
        batch_size=kwargs.get('test_batch_size'), data_source='test'),
        desc='testing')
    target_cm, sentiment_cm = np.zeros(
        (1, C_tar, C_tar), dtype=np.int32), np.zeros((1, C_sent, C_sent),
                                                     dtype=np.int32)
    for input_tuple in test_batch_generator:
        if kwargs.get('embedding') == 'fasttext':
            feed_dict = build_feed_dict(input_tuple,
                                        input_type='embedding')
        else:
            feed_dict = build_feed_dict(input_tuple, input_type='ids')
        tar_p, tar_r, tar_f1, sent_p, sent_r, sent_f1 = model.test_on_batch(
            feed_dict)
        temp_target_cm, temp_sentiment_cm = get_confusion_matrix(
            feed_dict, model, C_tar=C_tar, C_sent=C_sent)
        target_metrics, sentiment_metrics = get_computed_metrics(
            feed_dict, dataset, model, C_tar=C_tar, C_sent=C_sent)

        tar_acc, tar_p, tar_r, tar_f1, tar_cr = target_metrics['acc'], \
                                                target_metrics['results']['strict']['precision'], \
                                                target_metrics['results']['strict']['recall'], \
                                                target_metrics['results']['strict']['f1'], \
                                                target_metrics['CR']

        sent_acc, sent_p, sent_r, sent_f1, sent_cr = sentiment_metrics['acc'], \
                                                     sentiment_metrics['results']['strict']['precision'], \
                                                     sentiment_metrics['results']['strict']['recall'], \
                                                     sentiment_metrics['results']['strict']['f1'], \
                                                     sentiment_metrics['CR']

        tar_results, tar_results_by_tag = target_metrics['results'], target_metrics['results_by_tag']
        sent_results, sent_results_by_tag = sentiment_metrics['results'], sentiment_metrics['results_by_tag']

        target_cm = np.append(target_cm,
                              np.expand_dims(temp_target_cm, axis=0),
                              axis=0)
        sentiment_cm = np.append(sentiment_cm,
                                 np.expand_dims(temp_sentiment_cm, axis=0),
                                 axis=0)
        test_batch_generator.set_description(
            f'[Test][Entity]: a-{tar_acc:.3f}, p-{tar_p:.3f}, r-{tar_r:.3f}, f1-{tar_f1:.3f} [Relevance]: a-{sent_acc:.3f}, p-{sent_p:.3f}, r-{sent_r:.3f}, f1-{sent_f1:.3f}'
        )

    performance_dict['test_tar_acc'], performance_dict['test_tar_p'], \
    performance_dict['test_tar_r'], performance_dict['test_tar_f1'], performance_dict['test_tar_CR'], \
    performance_dict['test_tar_results'], performance_dict['test_tar_results_by_tag'], \
        = tar_acc, tar_p, tar_r, tar_f1, tar_cr, tar_results, tar_results_by_tag

    performance_dict['test_sent_acc'], performance_dict['test_sent_p'], \
    performance_dict['test_sent_r'], performance_dict['test_sent_f1'], performance_dict['test_sent_CR'], \
    performance_dict['test_sent_results'], performance_dict['test_sent_results_by_tag'], \
        = sent_acc, sent_p, sent_r, sent_f1, sent_cr, sent_results, sent_results_by_tag

    test_batch_generator.write(
        f'[Test][Entity][Report]:\n {tar_cr}')
    test_batch_generator.write(
        f'[Test][Relevance][Report]:\n {sent_cr}')
    test_batch_generator.write(
        f'[Test][Entity][CM]:\n {str(np.sum(target_cm, axis=0))}')
    test_batch_generator.write(
        f'[Test][Relevance][CM]:\n {str(np.sum(sentiment_cm, axis=0))}')

    performance_dict['note'] = NOTE
    performance_dict['Time'] = str(datetime.datetime.now())
    performance_dict['script_file'] = os.path.basename(__file__)
    new_performance_dict = {}
    for key, value in performance_dict.items():
        new_key = key.replace('_tar_', '_entity_').replace('_sent_', '_relevance_')
        if type(value) is np.int64:
            new_performance_dict[new_key] = int(value)
        else:
            new_performance_dict[new_key] = value
    with open(kwargs.get('performance_file'), 'a+') as outfile:
        outfile.write(json.dumps(new_performance_dict) + '\n')

    logger.info('Training finished.')


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)
