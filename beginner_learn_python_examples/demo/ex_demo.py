# -*- coding: utf-8 -*-
"""
 Created on 2021/4/5 12:44
 Author: Taosy.W
 E-mail: 1105936347@qq.com
 Zhihu : https://www.zhihu.com/people/1105936347
 Github: https://github.com/AFei19911012
-------------------------------------------------
"""

import sys
import tensorflow as tf
import cv2
from typing import List, Dict
from datetime import date
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI


tf.test.is_gpu_available
print(f'Python version: {sys.version}')
print(f'Tensorflow version: {tf.__version__}')
print(f'OpenCV version: {cv2.__version__}')

app = FastAPI()


@app.get("/")
def root():
    a = "a"
    b = "b" + a
    return {"hello world": b}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6666)
