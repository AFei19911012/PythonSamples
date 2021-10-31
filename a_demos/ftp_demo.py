# -*- coding: utf-8 -*-
"""
 Created on 2021/5/30 9:56
 Filename   : ex_ftp.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
from ftplib import FTP


def main(host='192.168.3.250', port=21, user='', passwd='', start_dir=''):
    ftp = FTP()
    ftp.connect(host=host, port=port)
    ftp.login(user=user, passwd=passwd)
    ftp.cwd(start_dir)
    L = ftp.nlst()
    print(L)


if __name__ == '__main__':
    main()
