import numpy as np
from utils import load
import os
import torch
from models import Tree_Net
from utils import Condition_Setter


def extract_rule(path_to_grammar, category_vocab):
    binary_rule = []
    unary_rule = []
    unk_idx = category_vocab.unk_index + 1

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    min_freq = 10
    for rule in data:
        tokens = rule.split()
        freq = int(tokens[0])
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]] + 1
            left_cat = category_vocab[tokens[4]] + 1
            right_cat = category_vocab[tokens[5]] + 1
            if freq >= min_freq and unk_idx not in [left_cat, right_cat, parent_cat]:
                binary_rule.append([left_cat, right_cat, parent_cat])
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]] + 1
            child_cat = category_vocab[tokens[4]] + 1
            if freq >= min_freq and unk_idx not in [parent_cat, child_cat]:
                unary_rule.append([parent_cat, child_cat])
    return binary_rule, unary_rule


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

path_to_grammar = PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar"

device = torch.device('cpu')

print('loading tree list...')
train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
print(train_tree_list.category_vocab['S[dcl]'])
NUM_VOCAB = len(train_tree_list.content_vocab)
NUM_CATEGORY = len(train_tree_list.category_vocab)

tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

embedding_weight = tree_net.embedding.weight
linear_weight = tree_net.linear.weight
linear_bias = tree_net.linear.bias
content_vocab = []
for k, v in train_tree_list.content_vocab.stoi.items():
    content_vocab.append([k, v])

binary_rule, unary_rule = extract_rule(path_to_grammar, train_tree_list.category_vocab)

np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/embedding_weight_GloVe_{}d.csv".format(
        condition.embedding_dim),
    embedding_weight.detach().numpy())
np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/linear_weight_GloVe_{}d.csv".format(
        condition.embedding_dim),
    linear_weight.detach().numpy())
np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/linear_bias_GloVe_{}d.csv".format(
        condition.embedding_dim),
    linear_bias.detach().numpy())
with open(PATH_TO_DIR + "Hol-CCG/data/parsing/content_vocab.txt", mode='w') as f:
    for info_list in content_vocab:
        i = 0
        for info in info_list:
            if i % 2 == 0:
                f.write(str(info))
                f.write(' ')
            else:
                f.write(str(info + 1))
            i += 1
        f.write('\n')
np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/binary_rule.txt", np.array(binary_rule) + 1,
           fmt='%d', header=str(len(train_tree_list.category_vocab) - 1), comments="")
np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/unary_rule.txt", np.array(unary_rule) + 1,
           fmt='%d', header=str(len(train_tree_list.category_vocab) - 1), comments="")