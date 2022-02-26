import os
import pickle
import time
import codecs
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
import logging

from datetime import timedelta
from logging import handlers

from src.utils.config import config


def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_seed(SEED):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def map_example_to_dict(input_ids, token_type_ids, label):
    return {
        'Input-Token': input_ids,
        'Input-Segment': token_type_ids,
    }, label


def get_time_idf(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return: 返回使用多长时间
    """
    end_time = time.time()
    time_idf = end_time - start_time
    return timedelta(seconds=int(round(time_idf)))


def save_pkl(path, obj, obj_name, use_bert=False):
    """
    保存数据
    :param path:
    :param obj:
    :param obj_name:
    :return:
    """
    log.info(f'{obj_name} save in {path}  use_bert {use_bert}')
    with codecs.open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path, obj_name):
    """
    加载数据
    :param path:
    :param obj_name:
    :return:
    """
    log.info(f'load {obj_name} in {path}')
    with codecs.open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger


def training_curve(loss, acc, val_loss=None, val_acc=None):
    """
    :param loss:
    :param acc:
    :param val_loss:
    :param val_acc:
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss, color='r', label='Training Loss')
    if val_loss is not None:
        ax[0].plot(val_loss, color='g', label='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(acc, color='r', label='Training Accuracy')
    if val_loss is not None:
        ax[1].plot(val_acc, color='g', label='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)
    plt.show()


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


ensure_dir(config.log_folder_path)

log = create_logger(config.log_path)
