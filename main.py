# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/24
# File Name: main
# Edit Author: lnest
# ------------------------------------
import argparse
import logging
from tools.data_util import process
from common.use_logging import enable_log


if __name__ == '__main__':
    # use logging
    enable_log(logging.DEBUG)

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-d', '--idx_path', default='data/new_data/')
    arg_parser.add_argument('--data_format', default='word')
    arg_parser.add_argument('--head_cnt', default=20, type=int)
    arg_parser.add_argument('--mode', default='train')
    arg_parser.add_argument('-l', '--log_level', default='DEBUG')
    arg_parser.add_argument('--step', default='prepro')
    args = arg_parser.parse_args()

    if args.step == 'prepro':
        process(args)