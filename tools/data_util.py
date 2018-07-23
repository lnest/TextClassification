# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/22:
# File Name: data_util
# Edit Author: lnest
# ------------------------------------
import os
import json
import pandas as pd

from time_eval import time_count

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '../data/new_data/')
DEFAULT_PATH = [DATA_PATH + './test_set.csv', DATA_PATH + './train_set.csv']


def explore_data(datas=DEFAULT_PATH, line_seperator=',', cont_sep=' ', target_index=0, has_header=True):
    """
    :param datas:
    :param line_seperator:
    :param cont_sep:
    :param target_index:
    :param has_header:
    :return:
    """
    df = dict()
    tf = dict()
    corpus_size = 0
    assert isinstance(datas, list), 'datas should be a list'
    for data in datas:
        for line in open(data):
            if corpus_size == 0 and has_header:
                has_header = False
                continue
            conts = line.strip().split(line_seperator)
            if target_index < len(conts):
                target_words = conts[target_index].strip().split(cont_sep)
                uniq_words = set(target_words)
                print('1', len(target_words))
                print('2', len(uniq_words))
                for word in target_words:
                    tf.setdefault(word, 0)
                    tf[word] += 1
                for word in uniq_words:
                    df.setdefault(word, 0)
                    df[word] += 1
                corpus_size += 1
    return tf, df, corpus_size


def show_data(tf, df, corpus_size):
    dataframe = pd.DataFrame({'tf': tf, 'df': df})
    dataframe = dataframe / corpus_size
    sorted_df = dataframe.sort_values(by=['df'], ascending=False)
    head_df = sorted_df.head(20)
    return head_df


def save_df(save_path, df):
    df.to_csv(save_path)


@time_count
def generate_wordmap(word_map_file, data_dir=DEFAULT_PATH):
    if not os.path.exists(data_dir):
        print('invalid annotated data directory: {}'.format(data_dir))
        return

    word2idx = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
    idx2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
    word_cnt = len(word2idx)

    for corpus_file in data_dir:
        with open(corpus_file, 'rb') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.decode('utf-8').strip()
                words = list(line)
                for word in words:
                    if word not in word2idx:
                        word2idx[word] = word_cnt
                        idx2word[word_cnt] = word
                        word_cnt += 1
    word_map = {'word2idx': word2idx, 'idx2word': idx2word}
    json.dump(word_map, open(word_map_file, 'w'), ensure_ascii=False)


def process():
    pass
