# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import shutil
import random

from tqdm import tqdm

min_frame_count = 100
train_size = 0.8

# Get filenames under directory
top_dir = 'D:/ParsingData/trainingData_v9/data(sample)/'
filenames = glob.glob(top_dir + '**/*.pkl', recursive=True)
filenames = [f for f in filenames if int(f.split('_')[-3]) >= min_frame_count]

# Copy files to separate directories
train_dir = 'D:/ParsingData/trainingData_v9/train/'
test_dir  = 'D:/ParsingData/trainingData_v9/test/'
for filename in tqdm(filenames):
    if random.random() <= train_size:
        shutil.copy2(filename, train_dir)
    else:
        shutil.copy2(filename, test_dir)
