# Should be run on a folder with following subdirectory structure
# Folders: 0, 1, 2, ... 9, *, 
# Dataset acquired from https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators

import os
from PIL import Image
import numpy as np
from pathlib import Path 



data_folder = os.path.join(os.path.dirname(__file__), "numbers_operations")
data_split_folder = os.path.join(os.path.dirname(__file__), "numbers_operations_split")

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# 16 classes: 0-9, +, -, *, /,  [, ] in that order

def charToIndex(c):
    dict = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4,
        '5' : 5,
        '6' : 6,
        '7' : 7,
        '8' : 8,
        '9' : 9,
        '+' : 10,
        '-' : 11,
        '*' : 12,
        '%' : 13,
        '[' : 14,
        ']' : 15
    }
    return dict[c]

data = [None] * 16
for i in range(16):
    data[i] = []


for folder in filter(lambda f : os.path.isdir(os.path.join(data_folder, f)), os.listdir(data_folder)):
    print(folder)
    for img_file in os.listdir(os.path.join(data_folder, folder)):
        img = np.array(Image.open(os.path.join(data_folder, folder, img_file)).convert('L'))
        img = [255 - x for x in img]
        data[charToIndex(folder)].append(img)

for x in data:
    print(len(x))


train_set = []
val_set = []
test_set = []

# Shuffle each class' data
for c in data:
    np.random.shuffle(c)

# Split each class
for i in range(16):
    test_index =  int((1 - TEST_SPLIT) * len(data[i]))
    val_index = int((1 - TEST_SPLIT - VAL_SPLIT) * len(data[i]))

    for j in range(0, val_index):
        train_set.append(np.append(data[i][j], [i]))
    for j in range(val_index, test_index):
        val_set.append(np.append(data[i][j], [i]))
    for j in range(test_index, len(data[i])):
        test_set.append(np.append(data[i][j], [i]))

# Shuffle all data
np.random.shuffle(train_set)
np.random.shuffle(val_set)
np.random.shuffle(test_set)

np.save(os.path.join(data_split_folder, "numbers_operations_train"), train_set)
np.save(os.path.join(data_split_folder, "numbers_operations_val"), val_set)
np.save(os.path.join(data_split_folder, "numbers_operations_test"), test_set)
