# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/7/29
# File Name: use_logging
# Edit Author: lnest
# ------------------------------------

import os
import logging
# from logging.handlers import TimedRotatingFileHandler

PID = os.getpid()
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))  # get absolutely directory of this file


def enable_log(level=logging.DEBUG):
    # console output
    logger = logging.getLogger()
    logger.setLevel(level)
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(levelname)-8s: %(filename)-15s:%(lineno)3d\t%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # output of log file
    # filename_directory = os.path.join(CURRENT_DIRECTORY, '../logs')   # directory log files saved in
    # if not os.path.exists(filename_directory):
    #     os.makedirs(filename_directory)
    #
    # filename = os.path.join(filename_directory, str(PID) + 'TextClassification.log')
    # filehandler = TimedRotatingFileHandler(filename, when='midnight', interval=1, backupCount=7)
    # filehandler.setLevel(logging.INFO)
    # filehandler.suffix = '%Y%m%d'
    # formatter = logging.Formatter(fmt='%(levelname)-8s: %(filename)-15s:%(lineno)3d\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # filehandler.setFormatter(formatter)
    # logger.addHandler(filehandler)
