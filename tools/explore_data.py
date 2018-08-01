# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/29
# File Name: explore_data
# Edit Author: lnest
# ------------------------------------

import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from time_eval import time_count

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
            if isinstance(label_index, int) and 0 <= label_index < len(conts):
                corpus.append({'para': target_words, 'lable': int(conts[label_index]), 'id': corpus_size})  # id starts with 1
            else:
                corpus.append({'para': target_words, 'id': corpus_size})
    return tf, df, corpus_size, corpus


def show_data(tf, df, corpus_size, head_num=20):
    dataframe = pd.DataFrame({'tf': tf, 'df': df})
    dataframe = dataframe / corpus_size
    sorted_df = dataframe.sort_values(by=['df'], ascending=False)
    head_df = sorted_df.head(head_num)
    return head_df


def save_df(save_path, df):
    df.to_csv(save_path)