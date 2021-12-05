# -*- coding: utf-8 -*-
"""
 Created on 2021/5/14 15:57
 Filename   : hashlib_demo.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""

# =======================================================
import hashlib

if __name__ == '__main__':
    md = hashlib.md5()   # bytes
    for idx in range(30):
        md.update(bytes('000000', encoding='utf-8'))  # encode
        re = md.hexdigest()
        print(re)

'''
670b14728ad9902aecba32e22fa4f6bd
35b9ab5a36f3234dd26db357fd4a0dc1
3ea032bf79e8c116b05f4698d5a8e044
660719b4a7591769583a7c8d20c6dfa4
1d0064395af3c745f6c3194e92373d7a
c5e0eb03cbb4bea95ce3f8f48fca77d5
f573e011b414bf3f9dd284f7dad29592
b505acf9fc996902b0c547a2abfc62b2
978b0444e93c5f7d714575f28a77dca1
5b19445b70b493c78f3bc06eb7962315
13afb640349976dcbd9f2f0cad138fe6
6790d5dc8b6c6db6ab22b41cc4f8040e
949c4ec88ae01d7abf7618859b141f0d
fd162f31aa22bcb60ee0d58703c08098
d0ebfbf12f2f02c3d13fa782e91dc100
b08c68ffb271b1f5d4fe3f4f782584c4
8ccc3a89f81483cbc6b11123973a23fe
3ef62231142490b4799e172d5b9f1687
bd113eff72150daaeab533ad2e87d080
470ba2ba894d31cab6a53f20be650bc6
00064fc0959506d0571d0d7a9ab5c773
c7adfadec568d2283e2f699bca4f7a56
65dd00b99021913708b0c6257a2903ba
fa38783fea08d4d7cec9ca37143799e6
2cb446c4b9803e889e98f596ba785588
503a5fb8a643805b8b277ebf6c737aec
6ea30dd25ff2184a748d238cf93f5e50
ec76280e8585e7c208fb9be46323196c
ed0ea80249bd66cc04d853b064b2e61c
87c8e4eb2f8bd558cf3635d101f0aecd
'''