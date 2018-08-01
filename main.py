# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/24
# File Name: main
# Edit Author: lnest
# ------------------------------------
import argparse
import logging
from tools.data_util import process
from common.use_logging import set_level


def get_log_level(level):
    level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARN': logging.WARNING,
                 'ERROR': logging.ERROR, 'FATAL': logging.FATAL}
    return level_map.get(level.upper(), logging.INFO)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-d', '--idx_path', default='data/new_data/idx')
    arg_parser.add_argument('--data_format', default='char')
    arg_parser.add_argument('--head_cnt', default=20, type=int)
    arg_parser.add_argument('--mode', default='train')
    arg_parser.add_argument('-l', '--log_level', default='DEBUG')
    arg_parser.add_argument('--step', default='prepro')
    args = arg_parser.parse_args()

    # set logging
    set_level(get_log_level(args.log_level))
    logger = logging.getLogger()

    if args.step == 'prepro':
        process(args)