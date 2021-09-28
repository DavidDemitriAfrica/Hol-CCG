from torch.nn.init import xavier_uniform_, kaiming_uniform_, orthogonal_
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from allennlp.modules.elmo import batch_to_ids
import torch
import torch.nn as nn
from operator import itemgetter
from utils import circular_correlation, single_circular_correlation


class Node:
    def __init__(self, node_info):
        if node_info[0] == 'True':
            self.is_leaf = True
        else:
            self.is_leaf = False
        self.self_id = int(node_info[1])
        if self.is_leaf:
            content = node_info[2]
            self.content = [self.convert_content(content)]
            self.category = node_info[3]
            self.ready = True
        else:
            self.category = node_info[2]
            self.num_child = int(node_info[3])
            self.ready = False
            if self.num_child == 1:
                self.child_node_id = int(node_info[4])
            else:
                self.left_child_node_id = int(node_info[4])
                self.right_child_node_id = int(node_info[5])

    def convert_content(self, content):
        if content == "-LRB-":
            content = "("
        elif content == "-LCB-":
            content = "{"
        elif content == "-RRB-":
            content = ")"
        elif content == "-RCB-":
            content = "}"
        elif r"\/" in content:
            content = content.replace(r"\/", "/")
        return content


class Tree:
    def __init__(self, self_id, node_list):
        self.self_id = self_id
        self.node_list = node_list

    def set_node_composition_info(self):
        self.composition_info = []
        while True:
            num_ready_node = 0
            for node in self.node_list:
                if node.ready:
                    num_ready_node += 1
                elif not node.is_leaf and not node.ready:
                    if node.num_child == 1:
                        child_node = self.node_list[node.child_node_id]
                        if child_node.ready:
                            node.content = child_node.content
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, child_node.self_id, 0])
                    else:  # when node has two children
                        left_child_node = self.node_list[node.left_child_node_id]
                        right_child_node = self.node_list[node.right_child_node_id]
                        if left_child_node.ready and right_child_node.ready:
                            node.content = left_child_node.content + right_child_node.content
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, left_child_node.self_id, right_child_node.self_id])
            if num_ready_node == len(self.node_list):
                break
        self.sentence = self.node_list[-1].content

    def set_original_position_of_leaf_node(self):
        self.original_pos = []
        node = self.node_list[-1]
        if node.is_leaf:
            node.original_pos = 0
            self.original_pos.append([node.self_id, node.original_pos])
        else:
            node.start_idx = 0
            node.end_idx = len(node.content)
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if child_node.is_leaf:
                    child_node.original_pos = child_node.start_idx
                    self.original_pos.append([child_node.self_id, child_node.original_pos])

            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if left_child_node.is_leaf:
                    left_child_node.original_pos = left_child_node.start_idx
                    self.original_pos.append(
                        [left_child_node.self_id, left_child_node.original_pos])
                if right_child_node.is_leaf:
                    right_child_node.original_pos = right_child_node.start_idx
                    self.original_pos.append(
                        [right_child_node.self_id, right_child_node.original_pos])

    def correct_parse(self, whole_category_vocab):
        correct_node_list = []
        top_node = self.node_list[-1]
        top_node.start_idx = 0
        top_node.end_idx = len(top_node.content)
        if not top_node.is_leaf:
            # for unk category
            if whole_category_vocab[top_node.category] == 0:
                correct_node_list.append((1, len(top_node.content) + 1, -1))
            else:
                correct_node_list.append(
                    (1, len(top_node.content) + 1, whole_category_vocab[top_node.category] + 1))
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if not child_node.is_leaf:
                    # for unk category
                    if whole_category_vocab[child_node.category] == 0:
                        correct_node_list.append(
                            (child_node.start_idx + 1,
                             child_node.end_idx + 1,
                             -1))
                    else:
                        correct_node_list.append(
                            (child_node.start_idx + 1,
                             child_node.end_idx + 1,
                             whole_category_vocab[child_node.category] + 1))
            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if not left_child_node.is_leaf:
                    # for unk category
                    if whole_category_vocab[left_child_node.category] == 0:
                        correct_node_list.append(
                            (left_child_node.start_idx + 1,
                             left_child_node.end_idx + 1,
                             -1))
                    else:
                        correct_node_list.append(
                            (left_child_node.start_idx + 1,
                             left_child_node.end_idx + 1,
                             whole_category_vocab[left_child_node.category] + 1))
                if not right_child_node.is_leaf:
                    # for unk category
                    if whole_category_vocab[right_child_node.category] == 0:
                        correct_node_list.append(
                            (right_child_node.start_idx + 1,
                             right_child_node.end_idx + 1,
                             -1))
                    else:
                        correct_node_list.append(
                            (right_child_node.start_idx + 1,
                             right_child_node.end_idx + 1,
                             whole_category_vocab[right_child_node.category] + 1))
        return correct_node_list

    def set_word_split(self, tokenizer):
        sentence = " ".join(self.sentence)
        tokens = tokenizer.tokenize(sentence)
        tokenized_pos = 0
        word_split = []
        for original_pos in range(len(self.sentence)):
            word = self.sentence[original_pos]
            length = 1
            while True:
                temp = tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length]).replace(" ", "")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        self.word_split = word_split
        return word_split


class Tree_List:
    def __init__(
            self,
            PATH_TO_DATA,
            word_category_vocab,
            phrase_category_vocab,
            device=torch.device('cpu')):
        self.word_category_vocab = word_category_vocab
        self.phrase_category_vocab = phrase_category_vocab
        self.device = device
        self.set_tree_list(PATH_TO_DATA)
        self.set_category_id(self.word_category_vocab, self.phrase_category_vocab)

    def set_tree_list(self, PATH_TO_DATA):
        self.tree_list = []
        tree_id = 0
        node_list = []
        with open(PATH_TO_DATA, 'r') as f:
            node_info_list = [node_info.strip() for node_info in f.readlines()]
        node_info_list = [node_info.replace(
            '\n', '') for node_info in node_info_list]
        for node_info in node_info_list:
            if node_info != '':
                node = Node(node_info.split())
                node_list.append(node)
            elif node_list != []:
                self.tree_list.append(Tree(tree_id, node_list))
                node_list = []
                tree_id += 1

    def set_category_id(self, word_category_vocab, phrase_category_vocab):
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    node.category_id = word_category_vocab[node.category]
                else:
                    node.category_id = phrase_category_vocab[node.category]
            tree.set_node_composition_info()
            tree.set_original_position_of_leaf_node()

    def set_info_for_training(self, tokenizer=None):
        self.num_node = []
        self.sentence_list = []
        self.label_list = []
        self.original_pos = []
        self.composition_info = []
        self.word_split = []
        for tree in self.tree_list:
            self.num_node.append(len(tree.node_list))
            if self.embedder == 'bert':
                self.sentence_list.append(" ".join(tree.sentence))
            elif self.embedder == 'elmo':
                self.sentence_list.append(tree.sentence)
            label_list = []
            for node in tree.node_list:
                label_list.append([node.category_id])
            self.label_list.append(label_list)
            self.original_pos.append(
                torch.tensor(
                    tree.original_pos,
                    dtype=torch.long,
                    device=self.device))
            self.composition_info.append(
                torch.tensor(
                    tree.composition_info,
                    dtype=torch.long,
                    device=self.device))
            if self.embedder == 'bert':
                self.word_split.append(tree.set_word_split(tokenizer))
            else:
                self.word_split.append([])
        self.sorted_tree_id = np.argsort(self.num_node)

    def make_shuffled_tree_id(self):
        shuffled_tree_id = []
        splitted = np.array_split(self.sorted_tree_id, 50)
        for id_list in splitted:
            np.random.shuffle(id_list)
            shuffled_tree_id.append(id_list)
        return np.concatenate(shuffled_tree_id)

    def make_batch(self, BATCH_SIZE=None):
        # make batch content id includes leaf node content id for each tree belongs to batch
        batch_num_node = []
        batch_sentence_list = []
        batch_label_list = []
        batch_original_pos = []
        batch_composition_info = []
        batch_word_split = []
        num_tree = len(self.tree_list)

        if BATCH_SIZE is None:
            batch_tree_id_list = list(range(num_tree))
            batch_num_node.append(
                list(itemgetter(*batch_tree_id_list)(self.num_node)))
            batch_sentence_list.append(list(itemgetter(*batch_tree_id_list)(self.sentence_list)))
            batch_label_list.append(list(itemgetter(
                *batch_tree_id_list)(self.label_list)))
            batch_original_pos.append(list(itemgetter(*batch_tree_id_list)(self.original_pos)))
            batch_composition_info.append(list(itemgetter(
                *batch_tree_id_list)(self.composition_info)))
            batch_word_split.append(list(itemgetter(
                *batch_tree_id_list)(self.word_split)))
        else:
            # shuffle the tree_id in tree_list
            shuffled_tree_id = self.make_shuffled_tree_id()
            for idx in range(0, num_tree - BATCH_SIZE, BATCH_SIZE):
                batch_tree_id_list = shuffled_tree_id[idx:idx + BATCH_SIZE]
                batch_num_node.append(
                    list(itemgetter(*batch_tree_id_list)(self.num_node)))
                batch_sentence_list.append(
                    list(
                        itemgetter(
                            *
                            batch_tree_id_list)(
                            self.sentence_list)))
                batch_label_list.append(list(itemgetter(
                    *batch_tree_id_list)(self.label_list)))
                batch_original_pos.append(list(itemgetter(*batch_tree_id_list)(self.original_pos)))
                batch_composition_info.append(list(itemgetter(
                    *batch_tree_id_list)(self.composition_info)))
                batch_word_split.append(list(itemgetter(
                    *batch_tree_id_list)(self.word_split)))
            # the part cannot devided by BATCH_SIZE
            batch_num_node.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.num_node)))
            batch_sentence_list.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(self.sentence_list)))
            batch_label_list.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.label_list)))
            batch_original_pos.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(self.original_pos)))
            batch_composition_info.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.composition_info)))
            batch_word_split.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(self.word_split)))

        for idx in range(len(batch_num_node)):
            composition_list = batch_composition_info[idx]
            # set mask for composition info in each batch
            max_num_composition = max([len(i) for i in composition_list])
            # make dummy compoisition info to fill blank in batch
            dummy_compositin_info = [
                torch.ones(
                    max_num_composition - len(i),
                    4,
                    dtype=torch.long,
                    device=self.device) * -1 for i in composition_list]
            batch_composition_info[idx] = torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(composition_list, dummy_compositin_info)])

        # return zipped batch information, when training, extract each batch from zip itteration
        return list(zip(
            batch_num_node,
            batch_sentence_list,
            batch_original_pos,
            batch_composition_info,
            batch_label_list,
            batch_word_split))

    def set_vector(self, tree_net):
        with tqdm(total=len(self.tree_list)) as pbar:
            pbar.set_description("setting vector...")
            for tree in self.tree_list:
                if self.embedder == 'bert':
                    sentence = [" ".join(tree.sentence)]
                    word_split = [tree.word_split]
                else:
                    sentence = [tree.sentence]
                    word_split = None
                packed_sequence = tree_net.embed(sentence, word_split=word_split)
                if tree_net.use_lstm:
                    bi_lstm_output = tree_net.bi_lstm(packed_sequence)[0]
                    transformed_rep, _ = tree_net.combine_foward_backward_rep(bi_lstm_output)
                else:
                    transformed_rep, _ = tree_net.unpack_bert_output(packed_sequence)
                for pos in tree.original_pos:
                    vector_list = transformed_rep[0]
                    node_id = pos[0]
                    original_pos = pos[1]
                    node = tree.node_list[node_id]
                    node.vector = torch.squeeze(vector_list[original_pos])
                for composition_info in tree.composition_info:
                    num_child = composition_info[0]
                    parent_node = tree.node_list[composition_info[1]]
                    if num_child == 1:
                        child_node = tree.node_list[composition_info[2]]
                        parent_node.vector = child_node.vector
                    else:
                        left_node = tree.node_list[composition_info[2]]
                        right_node = tree.node_list[composition_info[3]]
                        parent_node.vector = single_circular_correlation(
                            left_node.vector, right_node.vector)
                pbar.update(1)


class Tree_Net(nn.Module):
    def __init__(
            self,
            num_word_cat,
            num_phrase_cat,
            embedder,
            model,
            tokenizer=None,
            learn_embedder=True,
            use_lstm=True,
            embedding_dim=1024,
            word_dropout=0.2,
            phrase_dropout=0.2,
            device=torch.device('cpu')):
        super(Tree_Net, self).__init__()
        self.num_word_cat = num_word_cat
        self.num_phrase_cat = num_phrase_cat
        self.embedder = embedder
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim
        self.learn_embedder = learn_embedder
        self.use_lstm = use_lstm
        self.base_modules = []
        self.base_params = []
        if self.use_lstm:
            self.bi_lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True)
            self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.relu = nn.LeakyReLU()
            orthogonal_(self.bi_lstm.weight_ih_l0)
            orthogonal_(self.bi_lstm.weight_hh_l0)
            kaiming_uniform_(self.W1.weight)
            kaiming_uniform_(self.W2.weight)
            self.word_classifier = nn.Linear(self.hidden_dim, self.num_word_cat)
            self.phrase_classifier = nn.Linear(self.hidden_dim, self.num_phrase_cat)
            self.base_modules.append(self.bi_lstm)
            self.base_modules.append(self.W1)
            self.base_modules.append(self.W2)
        else:
            self.word_classifier = nn.Linear(self.embedding_dim, self.num_word_cat)
            self.phrase_classifier = nn.Linear(self.embedding_dim, self.num_phrase_cat)
        self.word_dropout = nn.Dropout(p=word_dropout)
        self.phrase_dropout = nn.Dropout(p=phrase_dropout)
        xavier_uniform_(self.word_classifier.weight)
        xavier_uniform_(self.phrase_classifier.weight)
        self.base_modules.append(self.word_classifier)
        self.base_modules.append(self.phrase_classifier)
        for module in self.base_modules:
            for params in module.parameters():
                self.base_params.append(params)
        self.base_params = iter(self.base_params)
        self.device = device

    # input batch as tuple of training info
    def forward(self, batch):
        num_node = batch[0]
        sentence = batch[1]
        original_pos = batch[2]
        composition_info = batch[3]
        batch_label = batch[4]
        if self.embedder == 'bert':
            word_split = batch[5]
        elif self.embedder == 'elmo':
            word_split = None
        if self.learn_embedder:
            packed_sequence = self.embed(sentence, word_split)
        else:
            with torch.no_grad():
                packed_sequence = self.embed(sentence, word_split)
        if self.use_lstm:
            bi_lstm_output = self.bi_lstm(packed_sequence)[0]
            encoded_rep, len_unpacked = self.combine_foward_backward_rep(bi_lstm_output)
        else:
            encoded_rep, len_unpacked = self.unpack_bert_output(packed_sequence)
        vector = self.set_leaf_node_vector(num_node, encoded_rep, len_unpacked, original_pos)
        composed_vector = self.compose(vector, composition_info)
        word_vector, phrase_vector, word_label, phrase_label = self.devide_word_phrase(
            composed_vector, batch_label, original_pos)
        word_output = self.word_classifier(self.word_dropout(word_vector))
        phrase_output = self.phrase_classifier(self.phrase_dropout(phrase_vector))
        return word_output, phrase_output, word_label, phrase_label

    # embedding word vector using bert or elmo
    def embed(self, sentence, word_split=None):
        if self.embedder == 'bert':
            input = self.tokenizer(
                sentence,
                padding=True,
                return_tensors='pt').to(self.device)
            output = self.model(**input).last_hidden_state[:, 1:-1]
            rep = []
            lengths = []
            for vector, info in zip(output, word_split):
                temp = []
                for start_idx, end_idx in info:
                    temp.append(torch.mean(vector[start_idx:end_idx], dim=0))
                rep.append(torch.stack(temp))
                lengths.append(len(temp))
            rep = pad_sequence(rep, batch_first=True)
            lengths = torch.tensor(lengths, device=torch.device('cpu'))

        elif self.embedder == 'elmo':
            input = batch_to_ids(sentence).to(self.device)
            output = self.model(input)
            rep = output['elmo_representations'][0]
            mask = output['mask'].to(torch.device('cpu'))
            lengths = torch.count_nonzero(mask, dim=1)
        packed_sequence = pack_padded_sequence(
            rep, lengths=lengths, batch_first=True, enforce_sorted=False)
        return packed_sequence

    # combine the output of bidirectional LSTM, the output of foward and backward LSTM
    def combine_foward_backward_rep(self, packed_sequence):
        seq_unapcked, len_unpacked = pad_packed_sequence(packed_sequence, batch_first=True)
        forward_rep = seq_unapcked[:, :, :self.hidden_dim]
        backward_rep = seq_unapcked[:, :, self.hidden_dim:]
        combined_rep = self.relu(self.W1(forward_rep) + self.W2(backward_rep))
        return combined_rep, len_unpacked

    # unpack packed bert representation
    def unpack_bert_output(self, packed_sequence):
        bert_rep, len_unpacked = pad_packed_sequence(packed_sequence, batch_first=True)
        return bert_rep, len_unpacked

    def set_leaf_node_vector(self, num_node, combined_rep, len_unpacked, original_pos):
        vector = torch.zeros(
            (len(num_node),
             torch.tensor(max(num_node)),
             self.hidden_dim), device=self.device)
        for idx in range(len(num_node)):
            batch_id = torch.tensor([idx for _ in range(len_unpacked[idx])])
            # target_id is node.self_id
            target_id = torch.squeeze(original_pos[idx][:, 0])
            # source_id is node.original_pos
            source_id = torch.squeeze(original_pos[idx][:, 1])
            vector[(batch_id, target_id)] = combined_rep[(batch_id, source_id)]
        return vector

    def compose(self, vector, composition_info):
        # itteration of composition
        for idx in range(composition_info.shape[1]):
            # the positional index where the composition info of one child is located in batch
            one_child_compositino_idx = torch.squeeze(
                torch.nonzero(composition_info[:, idx, 0] == 1))
            one_child_composition_info = composition_info[composition_info[:, idx, 0] == 1][:, idx]
            one_child_parent_idx = one_child_composition_info[:, 1]
            # the child node index of one child composition
            child_idx = one_child_composition_info[:, 2]
            child_vector = vector[(one_child_compositino_idx, child_idx)]
            vector[(one_child_compositino_idx, one_child_parent_idx)] = child_vector
            two_child_composition_idx = torch.squeeze(
                torch.nonzero(composition_info[:, idx, 0] == 2))
            two_child_composition_info = composition_info[composition_info[:, idx, 0] == 2][:, idx]
            if len(two_child_composition_info) != 0:
                two_child_parent_idx = two_child_composition_info[:, 1]
                # left child node index of two child composition
                left_child_idx = two_child_composition_info[:, 2]
                right_child_idx = two_child_composition_info[:, 3]
                left_child_vector = vector[(two_child_composition_idx, left_child_idx)]
                right_child_vector = vector[(two_child_composition_idx, right_child_idx)]
                composed_vector = circular_correlation(left_child_vector, right_child_vector)
                vector[(two_child_composition_idx, two_child_parent_idx)] = composed_vector
        return vector

    def devide_word_phrase(self, vector, batch_label, original_pos):
        word_vector = []
        phrase_vector = []
        word_label = []
        phrase_label = []
        for i in range(vector.shape[0]):
            word_idx = torch.zeros(len(batch_label[i]), dtype=torch.bool, device=self.device)
            word_idx[original_pos[i][:, 0]] = True
            phrase_idx = torch.logical_not(word_idx)
            word_vector.append(vector[i, :len(batch_label[i])][word_idx])
            phrase_vector.append(vector[i, :len(batch_label[i])][phrase_idx])
            word_label.append(
                torch.tensor(
                    batch_label[i],
                    dtype=torch.long,
                    device=self.device)[word_idx])
            phrase_label.append(
                torch.tensor(
                    batch_label[i],
                    dtype=torch.long,
                    device=self.device)[phrase_idx])
        word_vector = torch.cat(word_vector)
        phrase_vector = torch.cat(phrase_vector)
        word_label = torch.squeeze(torch.vstack(word_label))
        phrase_label = torch.squeeze(torch.vstack(phrase_label))
        return word_vector, phrase_vector, word_label, phrase_label

    def set_word_split(self, sentence):
        tokenizer = self.tokenizer
        sentence = " ".join(sentence)
        tokens = tokenizer.tokenize(sentence)
        tokenized_pos = 0
        word_split = []
        for original_pos in range(len(sentence.split())):
            word = sentence.split()[original_pos]
            length = 1
            while True:
                temp = tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length]).replace(" ", "")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        self.word_split = word_split
        return word_split

    def cal_word_vectors(self, sentence):
        word_split = self.set_word_split(sentence)
        packed_sequence = self.embed([" ".join(sentence)], [word_split])
        if self.use_lstm:
            bi_lstm_output = self.bi_lstm(packed_sequence)[0]
            encoded_rep, _ = self.combine_foward_backward_rep(bi_lstm_output)
        else:
            encoded_rep, _ = self.unpack_bert_output(packed_sequence)
        return encoded_rep[0]
