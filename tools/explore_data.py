# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/29
# File Name: explore_data
# Edit Author: lnest
# ------------------------------------

import os
import sys
import json
import logging
import pandas as pd
from tqdm import tqdm
from tools.time_eval import time_count

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '../data/new_data/')
DEFAULT_PATH = [DATA_PATH + './test_set.csv', DATA_PATH + './train_set.csv']
logger = logging.getLogger()


@time_count
def explore_data(data_path=DEFAULT_PATH[1], line_seperator=',', cont_sep=' ', target_index=0, label_index=None, has_header=True):
    """

    :param data_path: file_names will be explored
    :param line_seperator: field seperator
    :param cont_sep: sperator in every field
    :param target_index: field index
    :param label_index: 
    :param has_header: first line is header or not
    :return: 
    """
    df = dict()
    tf = dict()
    corpus_size = 0
    corpus = list()
    if not os.path.exists(data_path):
        logger.warning('path {} do not exsit!'.format(data_path))
        sys.exit(1)

    for line in tqdm(open(data_path).readlines()):

        # skip first line
        if corpus_size == 0 and has_header:
            has_header = False
            continue
        conts = line.strip().split(line_seperator)
        if target_index < len(conts):
            target_words = conts[target_index].strip().split(cont_sep)
            uniq_words = set(target_words)
            for word in target_words:
                tf.setdefault(word, 0)
                tf[word] += 1
            for word in uniq_words:
                df.setdefault(word, 0)
                df[word] += 1
            corpus_size += 1

            # train dataset has a lable, conversely test dataset is not
            if isinstance(label_index, int) and label_index < len(conts):
                corpus.append({'para': target_words, 'label': int(conts[label_index]), 'id': corpus_size})  # id starts with 1
            else:
                corpus.append({'para': target_words, 'id': corpus_size})
    return tf, df, corpus_size, corpus


def count_label(corpus):
    label_cnt = dict()
    for para in corpus:
        label = para.get('label')
        label_cnt.setdefault(label, 0)
        label_cnt[label] += 1
    return label_cnt


def show_data(tf, df, corpus_size, head_num=20):
    dataframe = pd.DataFrame({'tf': tf, 'df': df})
    dataframe = dataframe / corpus_size
    sorted_df = dataframe.sort_values(by=['df'], ascending=False)
    print('df tf info:\n', sorted_df.head(head_num))
    return sorted_df


def save_df(save_path, df):
    df.to_csv(save_path)


def get_stop_words(corpus_info, stop_words_path='./data/dinfo/stop_words_word', df_threshold_max=0.5, df_threshold_min=0.000010, refresh=False):
    if os.path.exists(stop_words_path) and refresh is False:
        logger.info('Stop words path exist. Set refresh true to regenerate word map')
        return set(json.load(open(stop_words_path)))
    df_upper = corpus_info[corpus_info['df'] > df_threshold_max]
    df_lower = corpus_info[corpus_info['df'] < df_threshold_min]
    stop_words = set(df_upper.index).union(df_lower.index)
    json.dump(list(stop_words), open(stop_words_path, 'w'))
    return stop_words


def explore(args):
    if not os.path.exists(args.dinfo_path):
        os.makedirs(args.dinfo_path)

    # get tf, df without normaliztion
    tf_word, df_word, corpus_size, corpus_word = explore_data(target_index=1, label_index=-1)
    word_corpus_info = show_data(tf_word, df_word, corpus_size, head_num=20)
    save_df(os.path.join(args.dinfo_path, 'word_corpus_info'), word_corpus_info)
    get_stop_words(word_corpus_info, os.path.join(args.dinfo_path, 'stop_words'))

    label_cnt = count_label(corpus_word)
    label_cnt.update({-1: corpus_size})
    json.dump(label_cnt, open(os.path.join(args.dinfo_path, 'label_cnt'), 'w'))
    series = pd.Series(label_cnt)
    print('label cnt info:\n', series.div(corpus_size).sort_index())

    tf_seg_word, df_seg_word, seg_corpus_size, corpus_seg = explore_data(target_index=2, label_index=-1)
    seg_corpus_info = show_data(tf_seg_word, df_seg_word, seg_corpus_size, head_num=20)
    save_df(os.path.join(args.dinfo_path, 'seg_corpus_info'), seg_corpus_info)
    get_stop_words(seg_corpus_info, os.path.join(args.dinfo_path, 'stop_words_segged'))