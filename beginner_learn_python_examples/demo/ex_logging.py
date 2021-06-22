# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:20
 Filename   : ex_logging.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
from typing import Optional
import logging
import sys
import time
from logging import handlers


def init_logger(name: str, filename: Optional[bool] = None, mode: str = 'w', stdout: bool = True, rotating_size: bool = False, rotating_time: bool = False):
    """
    日志模块
    :param name: 日志名称
    :param filename: 日志文件名
    :param mode: 写模式
    :param stdout: 是否终端输出
    :param rotating_size: 按文件大小重写
    :param rotating_time: 按日期重写
    :return:
    """
    if name in logging.Logger.manager.loggerDict.keys():
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s| %(module)-20s| %(funcName)-15s| %(lineno)-3d | %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]:")

    if stdout:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if not filename:
        filename = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()) + '.log'

    if rotating_time:
        # 每 1(interval) 天(when: Y/m/d/H/M/S) 重写 1 个文件，保留 3(backupCount) 个旧文件
        sh = handlers.TimedRotatingFileHandler(filename, when='d', interval=1, backupCount=3, encoding='UTF-8')
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if rotating_size:
        th = handlers.RotatingFileHandler(filename, mode=mode, maxBytes=1024*1024*5, backupCount=3, encoding='UTF-8')
        th.setLevel(logging.INFO)
        th.setFormatter(formatter)
        logger.addHandler(th)
    else:
        fh = logging.FileHandler(filename, mode=mode, encoding='UTF-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def main():
    logger = init_logger('MainTest', filename='logs_test.log')
    logger.info('init main ...')
    logger.error('error ...')
    logger.warning('warning ...')


if __name__ == '__main__':
    main()
