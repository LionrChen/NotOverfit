#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'ChenSir'
__mtime__ = '2019/7/12'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
            ┃       ┃
            ┃ ┳┛ ┗┳ ┃
            ┃   ┻   ┃
            ┗━┓   ┏━┛
              ┃   ┗━━━┓
              ┃神兽保佑 ┣┓
              ┃ 永无BUG！ ┏┛
              ┗┓┓┏━━┳┓┏┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

train[train.columns[2:]].std().plot('hist')
plt.title("Distribution of stds of all columns")
plt.show()