# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/22:
# File Name: data_util
# Edit Author: lnest
# ------------------------------------
import os
import sys
import json
import logging
from tqdm import tqdm
from time_eval import time_count

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '../data/new_data/')
DEFAULT_PATH = [DATA_PATH + './train_set.csv', DATA_PATH + './test_set.csv']
LOGGER = logging.getLogger()


@time_count
def read_corpus(corpus_path, line_seperator=',', cont_sep=' ', target_index=0, label_index=None, has_header=True):
    corpus_size = 0
    corpus = list()
    if not os.path.exists(corpus_path):
        LOGGER.warning('path {} do not exsit!'.format(corpus_path))
        sys.exit(1)

    for line in tqdm(open(corpus_path).readlines()):

        # skip first line
        if corpus_size == 0 and has_header:
            has_header = False
            continue
        conts = line.strip().split(line_seperator)
        if target_index < len(conts):
            target_words = conts[target_index].strip().split(cont_sep)
            corpus_size += 1

            # train dataset has a lable, conversely test dataset is not
            if isinstance(label_index, int) and 0 <= label_index < len(conts):
                corpus.append({'para': target_words, 'lable': int(conts[label_index]), 'id': corpus_size})  # id starts with 1
            else:
                corpus.append({'para': target_words, 'id': corpus_size})
    return corpus_size, corpus


@time_count
def generate_wordmap(word_map_file, data_dir=DEFAULT_PATH, refresh=False):
    if not os.path.exists(data_dir):
        print('invalid annotated data directory: {}'.format(data_dir))
        sys.exit(-1)

    if os.path.exists(word_map_file) and not refresh:
        LOGGER.info('File {} exists. Set refresh to true regenerate word map'.format(word_map_file))
        return json.load(word_map_file)

    word2idx = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
    idx2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
    word_cnt = len(word2idx)

    for corpus_file in data_dir:
        with open(corpus_file, 'rb') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip()
                words = list(line)
                for word in words:
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


def write_data(data, file_path):
    if data:
        f = open(file_path, 'w')
        f.write(data)
        f.close()
    else:
        LOGGER.info('EMPTY DATA!')


@time_count
def generate_dl_data(corpus, args, word_map, max_para_len=1000, dev_ratio=0.1):
    if word_map is None:
        LOGGER.info('word map is None!')
        return

    print('corpus size:', len(corpus))
    pivot = -1
    if args.mode == 'train':
        pivot = int(dev_ratio * len(corpus))

    dev_buffer = list()
    train_buffer = list()
    cnt = 1
    for sample in tqdm(corpus):
        if 'para' not in sample:
            continue
        para = sample.get('para')
        para = token2idx(para, word_map)
        ids = add_pad(para, max_para_len)
        sample['para'] = ids
        str_sample = json.dumps(sample)
        if pivot > 0 and cnt <= pivot:
            dev_buffer.append(str_sample)
        else:
            train_buffer.append(str_sample)
    train_data = '\n'.join(train_buffer)
    dev_data = '\n'.join(dev_buffer)
    write_data(train_data, '_'.join([args.idx_path, args.data_format, args.mode]))
    if args.mode == 'train':
        write_data(dev_data, '_'.join([args.idx_path, args.data_format, 'dev']))

    return len(train_buffer), len(dev_buffer)


def process(args):
    word_or_char = args.data_format
    word_map = generate_wordmap('token_map_' + word_or_char)

    is_train = True if args.mode == 'train' else False
    corpus_file = DEFAULT_PATH[0] if is_train else DEFAULT_PATH[1]

    if word_or_char == 'char':
        corpus_size, corpus = read_corpus(corpus_file, target_index=1)
    else:
        corpus_size, corpus = read_corpus(corpus_file, target_index=2)

    size = generate_dl_data(corpus, args, word_map)
    print(size)
