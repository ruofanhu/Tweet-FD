import sys
import collections
import numpy as np
import pandas as pd
import random
import math
import os
import gc
import json
from sklearn.model_selection import train_test_split

try:
    import ConfigParser as configparser
except:
    import configparser

from model import MLTModel
from evaluator import MLTEvaluator, MyEvaluator

NOTE = 'V0.0.1: initial version for mgade code'


def read_input_files(file_paths, max_sentence_length=-1):
    """
    Reads input files in whitespace-separated format.
    Will split file_paths on comma, reading from multiple files.
    """
    sentences = []
    line_length = None
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    assert (len(line_parts) >= 2), line
                    assert (len(line_parts) == line_length or line_length == None)
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                    sentences.append(sentence)
    return sentences


def convert_to_data(x_list, token_list, max_length=-1):
    sentences = []
    for x, t in zip(x_list, token_list):
        if max_length <= 0:
            sentence = [[i, j] for (i, j) in zip(x, t)]
        else:
            sentence = [[i, j] for (i, j) in zip(x[:max_length], t[:max_length])]
        sentences.append(sentence)
    return sentences


def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
    """
    Groups together sentences into batches
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    batches_of_sentence_ids = []
    if batch_equal_size == True:
        sentence_ids_by_length = collections.OrderedDict()
        sentence_length_sum = 0.0
        for i in range(len(sentences)):
            length = len(sentences[i])
            if length not in sentence_ids_by_length:
                sentence_ids_by_length[length] = []
            sentence_ids_by_length[length].append(i)

        for sentence_length in sentence_ids_by_length:
            if max_batch_size > 0:
                batch_size = max_batch_size
            else:
                batch_size = int((-1 * max_batch_size) / sentence_length)

            for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    else:
        current_batch = []
        max_sentence_length = 0
        for i in range(len(sentences)):
            current_batch.append(i)
            if len(sentences[i]) > max_sentence_length:
                max_sentence_length = len(sentences[i])
            if (0 < max_batch_size <= len(current_batch)) \
                    or (max_batch_size <= 0 and len(current_batch) * max_sentence_length >= (-1 * max_batch_size)):
                batches_of_sentence_ids.append(current_batch)
                current_batch = []
                max_sentence_length = 0
        if len(current_batch) > 0:
            batches_of_sentence_ids.append(current_batch)
    return batches_of_sentence_ids


def extract_from_dataframe(dataframe, columns, index):
    return_list = []
    for col in columns:
        return_list.append(dataframe[col].loc[index].to_numpy())
    return return_list


def process_sentences(data, y, model, is_training, learningrate, config, name):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    evaluator = MyEvaluator(config)
    batches_of_sentence_ids = create_batches_of_sentence_ids(data, config["batch_equal_size"], config["max_batch_size"])
    if is_training:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data[i] for i in sentence_ids_in_batch]
        y_batch = y[sentence_ids_in_batch]
        cost, sentence_scores, token_scores_list, token_probs, \
        token_probs_all_labels, unsup_weights, selective_weights, indicative_weights = model.process_batch(batch,
                                                                                                           y_batch,
                                                                                                           is_training,
                                                                                                           learningrate)

        evaluator.append_data(cost, batch, y_batch, sentence_scores, token_scores_list, token_probs)

        while config["garbage_collection"] == True and gc.collect() > 0:
            pass

    results = evaluator.get_results(name)
    for key in results:
        print(key + ": " + str(results[key]))
    return results


def run_experiment(config_path):
    config = parse_config("config", config_path)
    performance_dict = {k: v for (k, v) in list(config.items())}
    temp_model_path = config_path + ".model"
    performance_dict['temp_model_path'] = temp_model_path
    performance_dict['config_path'] = config_path
    random_seed = config['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    data_train, data_dev, data_test = None, None, None
    if config["path_data"] is not None and len(config["path_data"]) > 0:
        all_data = pd.read_pickle(config["path_data"])
        train_index, val_test_index = train_test_split(all_data.index, test_size=0.2, random_state=random_seed)
        val_index, test_index = train_test_split(val_test_index, test_size=0.5, random_state=random_seed)
        need_columns = ['tweet_tokens']
        if config["task_type"] == 'entity_detection':
            need_columns.append('entity_label')
        elif config["task_type"] == 'relevant_entity_detection':
            need_columns.append('relevant_entity_label')
        elif config["task_type"] == 'entity_relevance_classification':
            need_columns.append('relevance_entity_class_label')
        need_columns.append('sentence_class')
        X_train_raw, token_label_train_raw, y_train = extract_from_dataframe(all_data, need_columns, train_index)
        X_dev_raw, token_label_dev_raw, y_dev = extract_from_dataframe(all_data, need_columns, val_index)
        X_test_raw, token_label_test_raw, y_test = extract_from_dataframe(all_data, need_columns, test_index)
        data_train = convert_to_data(X_train_raw, token_label_train_raw, config["max_train_sent_length"])
        data_dev = convert_to_data(X_dev_raw, token_label_dev_raw)
        data_test = convert_to_data(X_test_raw, token_label_test_raw)
    else:
        if config["path_train"] is not None and len(config["path_train"]) > 0:
            data_train = read_input_files(config["path_train"], config["max_train_sent_length"])
            y_train = np.load(config["path_train_y"])
        if config["path_dev"] is not None and len(config["path_dev"]) > 0:
            data_dev = read_input_files(config["path_dev"])
            y_dev = np.load(config["path_dev_y"])
        if config["path_test"] is not None and len(config["path_test"]) > 0:
            data_test = []
            y_test_list = []
            for path_test, path_test_y in zip(config["path_test"].strip().split(":"),
                                              config["path_test_y"].strip().split(":")):
                data_test += read_input_files(path_test)
                y_test_list.append(np.load(path_test_y))
                y_test = np.concatenate(y_test_list, axis=0)

    model = MLTModel(config)
    model.build_vocabs(data_train, data_dev, data_test, config["preload_vectors"])
    model.construct_network()
    model.initialize_session()
    if config["preload_vectors"] is not None:
        model.preload_word_embeddings(config["preload_vectors"])

    print("parameter_count: " + str(model.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " + str(model.get_parameter_count_without_word_embeddings()))

    if data_train is not None:
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]
        for epoch in range(config["epochs"]):
            print("EPOCH: " + str(epoch))
            print("current_learningrate: " + str(learningrate))
            data_train = np.array(data_train, dtype=object)
            indices = np.arange(data_train.shape[0])
            np.random.shuffle(indices)
            data_train = data_train[indices].tolist()
            y_train = y_train[indices]

            results_train = process_sentences(data_train, y_train, model, is_training=True, learningrate=learningrate,
                                              config=config, name="train")

            if data_dev != None:
                dev_name = "valid"
                results_dev = process_sentences(data_dev, y_dev, model, is_training=False, learningrate=0.0,
                                                config=config, name=dev_name)

                if math.isnan(results_dev[f"{dev_name}_cost_sum"]) or math.isinf(results_dev[f"{dev_name}_cost_sum"]):
                    raise ValueError("Cost is NaN or Inf. Exiting.")

                if (epoch == 0 or (model_selector_type == "high" and results_dev[model_selector] > best_selector_value)
                        or (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev[model_selector]
                    performance_dict.update(results_dev)
                    performance_dict.update(results_train)
                    model.saver.save(model.session, temp_model_path,
                                     latest_filename=os.path.basename(temp_model_path) + ".checkpoint")
                print("best_epoch: " + str(best_epoch))

                if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config[
                    "stop_if_no_improvement_for_epochs"]:
                    break

                if (epoch - best_epoch) > 3:
                    learningrate *= config["learningrate_decay"]

            while config["garbage_collection"] == True and gc.collect() > 0:
                pass

        if data_dev is not None and best_epoch >= 0:
            # loading the best model so far
            model.saver.restore(model.session, temp_model_path)
            os.remove(temp_model_path + ".checkpoint")
            os.remove(temp_model_path + ".data-00000-of-00001")
            os.remove(temp_model_path + ".index")
            os.remove(temp_model_path + ".meta")

    if config["save"] is not None and len(config["save"]) > 0:
        model.save(config["save"])

    if config["path_data"] is not None:
        results_test = process_sentences(data_test, y_test, model, is_training=False, learningrate=0.0,
                                         config=config, name="test")
        performance_dict.update(results_test)
    elif config["path_test"] is not None:
        i = 0
        for path_test, path_test_y in zip(config["path_test"].strip().split(":"),
                                          config["path_test_y"].strip().split(":")):
            data_test = read_input_files(path_test)
            y_test = np.load(path_test_y)
            results_test = process_sentences(data_test, y_test, model, is_training=False, learningrate=0.0,
                                             config=config, name="test" + str(i))
            i += 1
            performance_dict.update(results_test)
    performance_dict['note'] = NOTE
    performance_dict['script_file'] = os.path.basename(__file__)
    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
    with open(config.get('performance_file'), 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')


if __name__ == "__main__":
    run_experiment(sys.argv[1])
