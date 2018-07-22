# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/22:
# File Name: data_util
# Edit Author: lnest
# ------------------------------------
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '../data/ccnews/')
DEFAULT_PATH = [DATA_PATH + './cnews.text.txt', DATA_PATH + './cnews.train.txt', DATA_PATH + './cnews.val.txt']


def explore_data(tf, df, datas=DEFAULT_PATH, line_seperator=',', cont_sep=' ', target_index=0, has_header=True):
    """

    :param tf:
    :param df:
    :param datas:
    :param line_seperator:
    :param cont_sep:
    :param target_index:
    :param has_header:
    :return:
    """
    corpus_size = 0
    assert isinstance(datas, list), 'datas should be a list'
    for data in datas:
        for line in open(data):
            if corpus_size == 0 and has_header:
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
                    tf[word] += 1
                corpus_size += 1
    return tf, df, corpus_size


def show_data(tf, df, corpus_size):
    dataframe = pd.DataFrame({'tf': tf, 'df': df})
    dataframe = dataframe / corpus_size
    sorted_df = dataframe.sort_values(by=['df'], ascending=False)
    sorted_df.head()

