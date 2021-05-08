import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt
from models import Tree_List, Tree_Net, Condition_Setter, History
from utils import load_weight_matrix, set_random_seed
from parsing import CCG_Category_List, Linear_Classifier, Parser
import time

PATH_TO_DIR = "/home/yamaki-ryosuke/Hol-CCG/"
condition = Condition_Setter(PATH_TO_DIR)

# initialize tree_list from toy_data
train_tree_list = Tree_List(
    condition.path_to_train_data, condition.REGULARIZED)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
# match the vocab and category between train and test data
test_tree_list.replace_vocab_category(train_tree_list)

train_tree_list.set_leaf_node_content_id()
batch_content_id = train_tree_list.make_batch(5)
a = 1
