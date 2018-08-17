# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/22:
# File Name: data_util
# Edit Author: lnest
# ------------------------------------
import os
import sys
import json
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from tools.time_eval import time_count

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '../data/new_data/')
DEFAULT_PATH = [DATA_PATH + './train_set.csv', DATA_PATH + './test_set.csv']
LOGGER = logging.getLogger()


@time_count
def read_corpus(corpus_path, line_seperator=',', cont_sep=' ', stop_words=None, segged_stop_words=None, article_index=0,
                segged_index=1, label_index=None, has_header=True):
    corpus_size = 0
    corpus = list()
    if not os.path.exists(corpus_path):
        LOGGER.warning('path {} do not exsit!'.format(corpus_path))
        sys.exit(1)

    if stop_words and segged_stop_words:
        LOGGER.info('length stop words: {}'.format(len(stop_words)))
        LOGGER.info('length segged stop words: {}'.format(len(segged_stop_words)))

    for line in tqdm(open(corpus_path).readlines()):

        article, segged_word = (None, None)
        # skip first line
        if corpus_size == 0 and has_header:
            has_header = False
            continue
        conts = line.strip().split(line_seperator)

        # get article and segged_words, then filt corpus using stop words
        if article_index < len(conts):
            article = conts[article_index].strip().split(cont_sep)
            if stop_words and isinstance(stop_words, set):
                article = [char for char in article if char not in stop_words]

        if segged_index < len(conts):
            segged_word = conts[segged_index].strip().split(cont_sep)
            if segged_stop_words and isinstance(segged_stop_words, set):
                segged_word = [word for word in segged_word if word not in segged_stop_words]

        if article and segged_word:
            # train dataset has a lable, conversely test dataset is not
            if isinstance(label_index, int) and label_index < len(conts):
                corpus.append({'para': article, 'segged_word': segged_word, 'lable': int(conts[label_index]), 'id': corpus_size})  # id starts with 1
            else:
                corpus.append({'para': article, 'segged_word': segged_word, 'id': corpus_size})
            corpus_size += 1
    return corpus_size, corpus


@time_count
def generate_wordmap(word_map_file, data_dir=DEFAULT_PATH, target_index=1, refresh=False, stop_words=None):
    for _dir in data_dir:
        if not os.path.exists(_dir):
            print('invalid annotated data directory: {}'.format(data_dir))
            sys.exit(-1)

    if os.path.exists(word_map_file) and not refresh:
        LOGGER.info('File {} exists. Set refresh true to regenerate word map'.format(word_map_file))
        return json.load(open(word_map_file))

    word2idx = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
    idx2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
    word_cnt = len(word2idx)

    for corpus_file in data_dir:
        with open(corpus_file, 'r') as fr:
            lines = fr.readlines()
            for line in tqdm(lines[1:]):
                line = line.strip().split(',')
                words = line[target_index]
                for word in words.split():
                    if word in stop_words:
                        continue
                    if word not in word2idx:
                        word2idx[word] = word_cnt
                        idx2word[word_cnt] = word
                        word_cnt += 1
    word_map = {'word2idx': word2idx, 'idx2word': idx2word}
    json.dump(word_map, open(word_map_file, 'w'), ensure_ascii=False)
    return word_map


def token2idx(tokens, word_map):
    if isinstance(word_map, dict) and 'word2idx' in word_map:
        word2idx = word_map.get('word2idx')
        return [word2idx.get(token, 3) for token in tokens]
    else:
        LOGGER.warning('INVALIDA WORD_MAP OBJECT!!')
        sys.exit(0)


def add_pad(ids, uniform_size):
    if len(ids) < uniform_size:
        for i in range(uniform_size - len(ids)):
            ids.append(0)
    else:
        ids = ids[:uniform_size]
    return ids


def corpus2idx(corpus, word_map, max_para_len=None, stop_words=None):
    if stop_words is not None:
        corpus = set(corpus).difference(stop_words)
    para = token2idx(corpus, word_map)
    if max_para_len:
        para = add_pad(para, max_para_len)
    return para


def write_data(data, file_path):
    if data:
        f = open(file_path, 'w')
        f.write(data)
        f.close()
    else:
        LOGGER.info('EMPTY DATA!')


def get_tf(article):
    tf = dict()
    for token in article:
        token = int(token)
        tf.setdefault(token, 0)
        tf[token] += 1
    tf_series = pd.Series(tf)
    # with TimeCountBlock('tf div article'):
    tf_series = tf_series.div(len(article))
    return tf_series


def get_corpus_feature(idf, article, token2idx_map, feature_len=100):
    """ get the first 100 words by tf-idf
    :return:
    """
    # with TimeCountBlock('get tf'):
    tf = get_tf(article)
    # with TimeCountBlock('get idf'):
    # LOGGER.debug('tf shape: {}'.format(tf.shape))
    # LOGGER.debug('idf shape: {}'.format(idf.shape))
    tf_idf = idf.mul(tf, fill_value=0)

    # with TimeCountBlock('sort_values'):
    feature_index = tf_idf.nlargest(feature_len).index
    filter_words = [word for word in article if word in feature_index]
    # with TimeCountBlock('corpus2idx'):
    filter_idx = corpus2idx(filter_words, token2idx_map, max_para_len=feature_len)
    return filter_idx


@time_count
def generate_dl_data(corpus, args, word_map, segged_map=None, idf=None, max_para_len=None, dev_ratio=0.1):
    if word_map is None:
        LOGGER.error('word map is None!')
        return

    LOGGER.debug('corpus size: {}'.format(len(corpus)))
    pivot = -1
    if args.mode == 'train':
        pivot = int(dev_ratio * len(corpus))

    dev_buffer = list()
    train_buffer = list()
    for cnt, sample in tqdm(enumerate(corpus, 1)):
        if 'para' not in sample or 'segged_word' not in sample:
            continue
        if idf is None:
            sample['para'] = corpus2idx(sample['para'], word_map, max_para_len)
            sample['segged_words'] = corpus2idx(sample['segged_word'], segged_map, max_para_len)
        else:
            sample['para'] = get_corpus_feature(idf.get('word'), sample['para'], word_map, max_para_len)
            sample['segged_word'] = get_corpus_feature(idf.get('segged'), sample['segged_word'], segged_map, max_para_len)
        str_sample = json.dumps(sample)
        if pivot > 0 and cnt <= pivot:
            dev_buffer.append(str_sample)
        else:
            train_buffer.append(str_sample)
        # if cnt >= 3:
        #     break
    train_data = '\n'.join(train_buffer)
    dev_data = '\n'.join(dev_buffer)
    write_data(train_data, '_'.join([args.idx_path, args.mode]))
    if args.mode == 'train':
        write_data(dev_data, '_'.join([args.idx_path, 'dev']))

    return len(train_buffer), len(dev_buffer)


def load_stop_words(stop_words_path):
    if os.path.exists(stop_words_path):
        stop_words = json.load(open(stop_words_path))
    else:
        LOGGER.error('Get stop words failed!')
        sys.exit(1)
    return set(stop_words)


def get_idf(corpus_info_path):
    df = pd.read_csv(corpus_info_path, index_col=0)
    idf = np.log(1 / df['df'])
    return idf


def process(args):

    # stop words
    segged_stop_words_path = os.path.join(args.dinfo_path, 'stop_words_segged')
    segged_stop_words = load_stop_words(segged_stop_words_path)
    stop_words_path = os.path.join(args.dinfo_path, 'stop_words')
    stop_words = load_stop_words(stop_words_path)

    # tokens map
    word_map = generate_wordmap('./data/maps/token_map_char', target_index=1, refresh=True, stop_words=stop_words)
    segged_map = generate_wordmap('./data/maps/token_map_word', target_index=2, refresh=True, stop_words=segged_stop_words)

    # corpus
    is_train = True if args.mode == 'train' else False
    corpus_file = DEFAULT_PATH[0] if is_train else DEFAULT_PATH[1]
    corpus_size, corpus = read_corpus(corpus_file, stop_words=stop_words, segged_stop_words=segged_stop_words, article_index=1, segged_index=2)

    # idf
    segged_idf = get_idf(os.path.join(args.dinfo_path, 'seg_corpus_info'))
    word_idf = get_idf(os.path.join(args.dinfo_path, 'word_corpus_info'))

    # use stop words filter
    print('before filt:', segged_idf.shape)
    keep_index_seg = set([str(row_label) for row_label in segged_idf.index]) - segged_stop_words
    segged_idf = segged_idf[segged_idf.index.isin(keep_index_seg)]
    print('After filt:', segged_idf.shape)

    print('before filt:', word_idf.shape)
    keep_index = set([str(row_label) for row_label in word_idf.index]) - stop_words
    word_idf = word_idf[word_idf.index.isin(keep_index)]
    idf = {'word': word_idf, 'segged': segged_idf}
    print('After filt:', word_idf.shape)

    size = generate_dl_data(corpus, args, word_map, segged_map, idf=idf, max_para_len=100)
    print(size)
