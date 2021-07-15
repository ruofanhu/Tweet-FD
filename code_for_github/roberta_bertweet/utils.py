from __future__ import print_function
import seqeval.metrics
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from nervaluate import Evaluator
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import six


def multi_batch_generator(X, y, token_label, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()

    X_copy = X_copy[indices]

    if y is not None:
        y_copy = y.copy()
        y_copy = y_copy[indices]

    if token_label is not None:
        token_label_copy = token_label.copy()
        token_label_copy = token_label_copy[indices]

    if masks is not None:
        masks_copy = masks.copy()
        masks_copy = masks_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            X_batch = X_copy[i:i + batch_size]
            y_batch = y_copy[i:i + batch_size] if y is not None else None
            t_batch = token_label_copy[i:i + batch_size] if token_label is not None else None
            m_batch = masks_copy[i:i + batch_size] if masks is not None else None
            yield X_batch, y_batch, t_batch, m_batch
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            token_label_copy = token_label_copy[indices]
            if masks is not None:
                masks_copy = masks_copy[indices]
            continue


def multi_batch_seq_generator(X, y, token_label, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]

    i = 0
    while True:
        if masks is not None:
            yield X[i:i + batch_size], y[i:i + batch_size], token_label[i:i + batch_size], masks[i:i + batch_size]
        else:
            yield X[i:i + batch_size], y[i:i + batch_size], token_label[i:i + batch_size]
        if i + batch_size >= size:
            break
        else:
            i += batch_size


def batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


def san_batch_generator(X, y, attention_labels, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()
    X_copy = X_copy[indices]

    y_copy = y.copy()
    y_copy = y_copy[indices]

    if attention_labels is not None:
        attention_labels_copy = attention_labels.copy()
        attention_labels_copy = attention_labels_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            if attention_labels is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size]
            else:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            if attention_labels is not None:
                attention_labels_copy = attention_labels_copy[indices]
            continue


def great_batch_generator(X, y, attention_labels, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()
    X_copy = X_copy[indices]

    if y is not None:
        y_copy = y.copy()
        y_copy = y_copy[indices]

    if attention_labels is not None:
        attention_labels_copy = attention_labels.copy()
        attention_labels_copy = attention_labels_copy[indices]

    if masks is not None:
        masks_copy = masks.copy()
        masks_copy = masks_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            if y is not None and attention_labels is not None and masks is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], attention_labels_copy[
                                                                          i:i + batch_size], masks_copy[
                                                                                             i:i + batch_size]
            elif y is None and attention_labels is not None and masks is not None:
                yield X_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size], masks_copy[i:i + batch_size]
            elif y is None and attention_labels is not None and masks is None:
                yield X_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size]
            elif y is not None and attention_labels is not None and masks is None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size]
            elif y is not None and attention_labels is None and masks is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], masks_copy[i:i + batch_size]
            elif y is not None and attention_labels is None and masks is None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            else:
                yield X_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            if y is not None:
                y_copy = y_copy[indices]
            if attention_labels is not None:
                attention_labels_copy = attention_labels_copy[indices]
            if masks is not None:
                masks_copy = masks_copy[indices]
            continue


def mask_batch_generator(X, y, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()

    X_copy = X_copy[indices]

    y_copy = y.copy()

    y_copy = y_copy[indices]

    if masks is not None:
        masks_copy = masks.copy()
        masks_copy = masks_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            if masks is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], masks_copy[i:i + batch_size]
            else:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            if masks is not None:
                masks_copy = masks_copy[indices]
            continue


def mask_batch_seq_generator(X, y, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]

    i = 0
    while True:
        if masks is not None:
            yield X[i:i + batch_size], y[i:i + batch_size], masks[i:i + batch_size]
        else:
            yield X[i:i + batch_size], y[i:i + batch_size]
        if i + batch_size >= size:
            break
        else:
            i += batch_size


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def align_predictions(predictions, label_ids, label_map):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    preds = preds.tolist()
    label_map_switch = {label_map[k]: k for k in label_map}
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i][j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map_switch[label_ids[i][j].item()])
                preds_list[i].append(label_map_switch[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(predictions, label_ids, label_map):
    labels = list(label_map.keys())
    labels = [i[2:] for i in labels if i.startswith('B-')]
    preds_list, out_label_list = align_predictions(predictions, label_ids, label_map)
    evaluator = Evaluator(out_label_list, preds_list, tags=labels, loader="list")
    results, results_by_tag = evaluator.evaluate()
    try:
        cls_report = seqeval.metrics.classification_report(out_label_list, preds_list, zero_division=1)
    except:
        cls_report = ""
    return {
        "accuracy_score": seqeval.metrics.accuracy_score(out_label_list, preds_list),
        "results": results,
        "results_by_tag": results_by_tag,
        "CR": cls_report,
    }


def align_predictions_crf(predictions, tag_ids, label_map):
    # tag_ids is a list of lists the true label: [[1,3,4],[3,5,6,7,8],[0,3]]
    label_map_i = {y: x for x, y in label_map.items()}
    batch_size = len(predictions)  # prediction: a list of lists with different length
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        out_label_list[i].extend([label_map_i[x] for x in tag_ids[i]])
        preds_list[i].extend([label_map_i[x] for x in predictions[i]])

    return preds_list, out_label_list


def compute_crf_metrics(predictions, label_ids, label_map):
    labels = list(label_map.keys())
    labels = [i[2:] for i in labels if i.startswith('B-')]
    preds_list, out_label_list = align_predictions_crf(predictions, label_ids, label_map)
    evaluator = Evaluator(out_label_list, preds_list, tags=labels, loader="list")
    results, results_by_tag = evaluator.evaluate()
    try:
        cls_report = seqeval.metrics.classification_report(out_label_list, preds_list, zero_division=1)
    except:
        cls_report = ""
    return {
        "accuracy_score": seqeval.metrics.accuracy_score(out_label_list, preds_list),
        "results": results,
        "results_by_tag": results_by_tag,
        "CR": cls_report,
    }


def read_txtfile_tolist(filepath):
    all_text = []
    all_labels = []
    orig_tokens = []
    orig_labels = []

    file = open(filepath, "rt")
    d = file.readlines()
    file.close()
    for line in d:
        line = line.rstrip()

        if not line:
            all_text.append(orig_tokens)
            all_labels.append(orig_labels)
            orig_tokens = []
            orig_labels = []
        else:
            token, label = line.split()
            orig_tokens.append(token)
            orig_labels.append(label)

    return all_text, all_labels


def extract_from_dataframe(dataframe, columns):
    return_list = []
    for col in columns:
        return_list.append(dataframe[col].to_numpy())
    return return_list


def new_build_glove_embedding(embedding_path= 'embeddings/glove.840B.300d.txt' ):
    '''
    PAD_INDEX: 2196016
    UNK_INDEX: 2196017
    '''
    embeddings_index = dict()
    embeddings = []
    with open(embedding_path) as f:
        for idx, line in enumerate(f):
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings.append(coefs)
            embeddings_index[word] = idx
    embeddings = np.array(embeddings)
    pad = np.zeros(shape=(1, embeddings.shape[1]))
    unk = np.random.uniform(-0.25, 0.25, size=(1, embeddings.shape[1]))
    embeddings = np.concatenate((embeddings, pad, unk), axis=0)
    return embeddings_index, embeddings




if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array(['a', 'b', 'c', 'd']), np.array([1, 2, 3, 4]), 2)
    for _ in range(8):
        xx, yy = next(gen)
        print(xx, yy)
