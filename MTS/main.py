# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:59:25 2018

@author: danil
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
import osmread

from tqdm import tqdm_notebook
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import os


#размер графиков
mpl.rcParams['figure.figsize'] = (8, 7)
#стиль и размер шрифта
sns.set(style='whitegrid', font_scale=1.5)