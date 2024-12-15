from torchtext.vocab import vocab, build_vocab_from_iterator
from collections import Counter
import random
from tqdm import tqdm
import numpy as np
import torch
from operator import itemgetter
from utils import circular_correlation, circular_convolution, convert_content
from typing import List, Union
from transformers import RobertaTokenizer, BertTokenizer
from holccg import HolCCG


class Node:
    def __init__(self, node_info: list) -> None:
        """
        node_info format:
        [is_leaf (True/False), self_id, content/category, category_if_leaf (if leaf), pos_if_leaf (if leaf), ...]
        """
        self.self_id = int(node_info[1])
        if node_info[0] == 'True':
            self.is_leaf = True
            # For leaf: store just the single word and its category
            self.word = convert_content(node_info[2])
            self.category = node_info[3]
            self.pos = node_info[4]
            # Will set indices later
            self.start_idx = None
            self.end_idx = None
            self.ready = True
        else:
            self.is_leaf = False
            self.category = node_info[2]
            self.num_child = int(node_info[3])
            if self.num_child == 1:
                self.child_node_id = int(node_info[4])
                self.head = int(node_info[5])
            else:
                self.left_child_node_id = int(node_info[4])
                self.right_child_node_id = int(node_info[5])
                self.head = int(node_info[6])
            # Will set indices once children are processed
            self.start_idx = None
            self.end_idx = None
            self.ready = False

class Tree:
    def __init__(self, self_id: int, node_list: list) -> None:
        self.self_id = self_id
        self.node_list = node_list

    def set_node_composition_info(self) -> None:
        # First pass: mark leaves as ready
        for node in self.node_list:
            if node.is_leaf:
                node.ready = True
            else:
                node.ready = False

        # Compute composition info bottom-up without storing full node.content arrays
        # Instead, we will just record composition steps and set start/end indices.
        self.composition_info = []
        # Determine the start/end indices for leaf nodes directly:
        # We'll do this after we figure out the sentence ordering.

        # To find sentence words: gather all leaves and their words
        # Leaves are in the order they appear in self.node_list, but we must ensure they're in sentence order.
        # We can rely on topological logic: The final node is the root. We'll assign start/end indices bottom-up.
        
        # Step 1: Gather words from leaf nodes in original order:
        # Initially, we don't know the original word order. We'll figure it out after composition.
        # Let's do a top-down pass to assign start/end indices:
        # At the end, the root node (last in list) covers the entire sentence.
        
        # A safe approach: Once composition_info is complete, the last node will cover the entire sentence.
        # So we first do a loop like before but without storing large arrays.

        # We'll emulate the original logic but only store minimal info:
        while True:
            num_ready_node = sum(node.ready for node in self.node_list)
            for node in self.node_list:
                if not node.ready and not node.is_leaf:
                    if node.num_child == 1:
                        child = self.node_list[node.child_node_id]
                        if child.ready:
                            # Parent inherits child's span
                            node.start_idx = child.start_idx
                            node.end_idx = child.end_idx
                            node.ready = True
                            self.composition_info.append([1, node.self_id, child.self_id, 0])
                    else:
                        left_child = self.node_list[node.left_child_node_id]
                        right_child = self.node_list[node.right_child_node_id]
                        if left_child.ready and right_child.ready:
                            # Parent span is combination of children
                            node.start_idx = left_child.start_idx
                            node.end_idx = right_child.end_idx
                            node.ready = True
                            self.composition_info.append([2, node.self_id, left_child.self_id, right_child.self_id])
            
            if num_ready_node == len(self.node_list):
                break

        # Now, all nodes have start/end indices.
        # The root node covers the entire sentence: from 0 to end_idx
        # Next, we must form the actual sentence. The leaves are the ultimate words:
        leaves = [n for n in self.node_list if n.is_leaf]
        # Sort leaves by position or assign their start_idx
        # If leaves were not assigned start_idx/end_idx yet, do so:
        # Each leaf is a single word, so start_idx == end_idx for leaves.
        # Let's assign them in the order they appear to form the sentence:
        # The original code expects self.sentence = a list of words in order.
        # In this approach, we have no actual "positions" yet. We must define them:
        # We'll assign leaf positions in order of appearance for simplicity:
        sentence_words = [leaf.word for leaf in leaves]
        self.sentence = sentence_words

        # Assign leaf indices now based on their order in the final sentence
        # Leaf i in leaves corresponds to sentence index i
        for i, leaf in enumerate(leaves):
            leaf.start_idx = i
            leaf.end_idx = i + 1

        # We must redo a pass to fix parent start/end indices since now leaves have final positions
        # Reset readiness and recompute:
        for node in self.node_list:
            if node.is_leaf:
                node.ready = True
            else:
                node.ready = False
        self.composition_info = []
        while True:
            num_ready_node = sum(node.ready for node in self.node_list)
            for node in self.node_list:
                if not node.ready and not node.is_leaf:
                    if node.num_child == 1:
                        child = self.node_list[node.child_node_id]
                        if child.ready:
                            node.start_idx = child.start_idx
                            node.end_idx = child.end_idx
                            node.ready = True
                            self.composition_info.append([1, node.self_id, child.self_id, 0])
                    else:
                        left_child = self.node_list[node.left_child_node_id]
                        right_child = self.node_list[node.right_child_node_id]
                        if left_child.ready and right_child.ready:
                            node.start_idx = left_child.start_idx
                            node.end_idx = right_child.end_idx
                            node.ready = True
                            self.composition_info.append([2, node.self_id, left_child.self_id, right_child.self_id])
            if num_ready_node == len(self.node_list):
                break

    def set_original_position_of_leaf_node(self) -> None:
        self.original_position = []
        self.spans = []

        # For leaves, original_position is just (node_id, start_idx)
        for node in self.node_list:
            if node.is_leaf:
                self.original_position.append([node.self_id, node.start_idx])

        # For spans, we record [start_idx, end_idx]
        for info in self.composition_info:
            if info[0] == 2:
                parent_node = self.node_list[info[1]]
                self.spans.append([parent_node.start_idx, parent_node.end_idx])

    def generate_random_tree(self) -> None:
        # This function can remain mostly the same, but we don't store node.content arrays anymore.
        # We only have start and end indices. 
        # Since the sentence is known (self.sentence), length = len(self.sentence)
        length = len(self.sentence)
        random_composition_info = []
        random_original_position = []
        negative_node_id = []
        node_id = 0

        node = [0, length, node_id]
        node_list = [node]

        if length > 1:
            if node[:2] not in self.spans:
                negative_node_id.append(node_id)
            wait_list = [node]
        else:
            wait_list = []
            random_original_position = [[0, 0]]

        node_id += 1

        while wait_list:
            parent_node = wait_list.pop(0)
            start_idx = parent_node[0]
            end_idx = parent_node[1]
            parent_id = parent_node[2]
            # decide split point
            split_idx = random.randint(start_idx + 1, end_idx - 1)

            left_node = [start_idx, split_idx, node_id]
            node_list.append(left_node)
            if (split_idx - start_idx) > 1:
                if left_node[:2] not in self.spans:
                    negative_node_id.append(node_id)
                wait_list.append(left_node)
            else:
                random_original_position.append([node_id, split_idx - 1])
            node_id += 1

            right_node = [split_idx, end_idx, node_id]
            node_list.append(right_node)
            if (end_idx - split_idx) > 1:
                if right_node[:2] not in self.spans:
                    negative_node_id.append(node_id)
                wait_list.append(right_node)
            else:
                random_original_position.append([node_id, end_idx - 1])
            node_id += 1

            random_composition_info.append([2, parent_id, node_id - 2, node_id - 1])

        random_composition_info.reverse()
        return len(node_list), random_composition_info, random_original_position, negative_node_id

    def set_word_split(self, tokenizer) -> list:
        # This can remain mostly unchanged, but we can directly work on self.sentence
        sentence_str = " ".join(self.sentence)
        tokens = tokenizer.tokenize(sentence_str)
        tokenized_pos = 0
        word_split = []
        for word in self.sentence:
            length = 1
            while True:
                temp = tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length])
                temp = temp.replace(" ", "").replace("\"", "``")
                if word.lower() == temp or word == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        self.word_split = word_split
        return word_split


class TreeList:
    def __init__(
            self,
            path_to_tree_list: str,
            type: str,
            word_category_vocab: vocab = None,
            phrase_category_vocab: vocab = None,
            head_info: dict = None,
            min_word_category: int = 0,
            min_phrase_category: int = 0,
            device: torch.device = None) -> None:
        """class for tree list

        Parameters
        ----------
        path_to_tree_list : str
            path to preprocessd tree list
        type : str
            type of tree list, 'train' or 'dev' or 'test'
        word_category_vocab : vocab, optional
            vocabulary for word category, by default None
        phrase_category_vocab : vocab, optional
            vocabulary for phrase category, by default None
        head_info : dict, optional
            dictionary of head information, by default None
        min_word_category : int, optional
            minimum frequency of word category, by default 0
        min_phrase_category : int, optional
            minimum frequency of phrase category, by default 0
        device : torch.device, optional
            device to use, by default None
        """
        # Handle device selection more robustly
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.type = type
        self.min_word_category = min_word_category
        self.min_phrase_category = min_phrase_category
        self.set_tree_list(path_to_tree_list)
        
        if type == 'train':
            self.set_vocab_and_head()
        else:
            self.word_category_vocab = word_category_vocab
            self.phrase_category_vocab = phrase_category_vocab
            self.head_info = head_info
        
        self.set_category_id()

    def set_tree_list(self, path_to_tree_list: str) -> None:
        """Construct the list of trees.

        Parameters
        ----------
        path_to_tree_list : str
            path to preprocessd tree list
        """
        self.tree_list = []
        tree_id = 0
        node_list = []
        with open(path_to_tree_list, 'r') as f:
            node_info_list = [node_info.strip() for node_info in f.readlines()]
        node_info_list = [node_info.replace(
            '\n', '') for node_info in node_info_list]
        with tqdm(total=len(node_info_list), unit="node_info") as pbar:
            pbar.set_description("Constructing tree list...")
            for node_info in node_info_list:
                if node_info != '':
                    node = Node(node_info.split())
                    node_list.append(node)
                elif node_list != []:
                    self.tree_list.append(Tree(tree_id, node_list))
                    node_list = []
                    tree_id += 1
                pbar.update(1)

    def set_vocab_and_head(self) -> None:
        """Construct vocabulary for word and phrase category and head information.
        """
        word_category_counter = Counter()
        phrase_category_counter = Counter()
        head_info_temp = {}
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    word_category_counter[node.category] += 1
                else:
                    if node.num_child == 2:
                        left_child = tree.node_list[node.left_child_node_id]
                        right_child = tree.node_list[node.right_child_node_id]
                        left = left_child.category.split('-->')[-1]
                        right = right_child.category.split('-->')[-1]
                        parent = node.category.split('-->')[0]
                        rule = (left, right, parent)
                        if rule not in head_info_temp:
                            head_info_temp[rule] = [0, 0]
                        head_info_temp[rule][node.head] += 1
                    phrase_category_counter[node.category] += 1
        self.word_category_vocab = build_vocab_from_iterator(
            word_category_counter,
            min_freq=self.min_word_category,
            specials=["<unk>"])
        self.phrase_category_vocab = build_vocab_from_iterator(
            phrase_category_counter,
            min_freq=self.min_phrase_category,
            specials=["<unk>"])
        self.head_info = {}
        for k, v in head_info_temp.items():
            # when left head is majority
            if v[0] >= v[1]:
                self.head_info[k] = 0
            # when right head is majority
            else:
                self.head_info[k] = 1

    def set_category_id(self) -> None:
        """Set category id for each node.
        """
        word_category_vocab = self.word_category_vocab
        phrase_category_vocab = self.phrase_category_vocab
        
        for tree in self.tree_list:
            for node in tree.node_list:
                try:
                    if node.is_leaf:
                        # Use get method with a default 'unk' index to handle unknown tokens
                        node.category_id = word_category_vocab[node.category]
                    else:
                        node.category_id = phrase_category_vocab[node.category]
                except (KeyError, RuntimeError):
                    # If token is not in vocab, assign the UNK token index
                    if node.is_leaf:
                        node.category_id = word_category_vocab['<unk>']
                    else:
                        node.category_id = phrase_category_vocab['<unk>']
            
            tree.set_node_composition_info()
            tree.set_original_position_of_leaf_node()

    def set_info_for_training(self, tokenizer: Union[RobertaTokenizer, BertTokenizer]) -> None:
        """Set information for training.

        Parameters
        ----------
        tokenizer : Union[RobertaTokenizer, BertTokenizer]
            tokenizer to use for tokenization
        """
        self.num_node = []
        self.sentence_list = []
        self.label_list = []
        self.original_position = []
        self.composition_info = []
        self.word_split = []
        for tree in self.tree_list:
            self.num_node.append(len(tree.node_list))
            self.sentence_list.append(" ".join(tree.sentence))
            label_list = []
            for node in tree.node_list:
                label_list.append([node.category_id])
            self.label_list.append(label_list)
            self.original_position.append(
                torch.tensor(
                    tree.original_position,
                    dtype=torch.long,
                    device=self.device))
            self.composition_info.append(
                torch.tensor(
                    tree.composition_info,
                    dtype=torch.long,
                    device=self.device))
            self.word_split.append(tree.set_word_split(tokenizer))
        self.sorted_tree_id = np.argsort(self.num_node)

    def make_shuffled_tree_id(self) -> np.ndarray:
        """Make shuffled tree id."""
        shuffled_tree_id = []
        splitted = np.array_split(self.sorted_tree_id, 4)
        for id_list in splitted:
            np.random.shuffle(id_list)
            shuffled_tree_id.append(id_list)
        return np.concatenate(shuffled_tree_id)

    def make_batch(self, batch_size: int = None) -> None:
        """Make batch for training.

        Parameters
        ----------
        batch_size : int, optional
            batch size, by default None
        """
        # make batch content id includes leaf node content id for each tree belongs to batch
        batch_num_node = []
        batch_sentence_list = []
        batch_label_list = []
        batch_original_position = []
        batch_composition_info = []
        batch_word_split = []
        num_tree = len(self.tree_list)

        # the series of "random" are information about randomly generated tree for
        # the scoring of span
        random_num_node = []
        random_original_position = []
        random_composition_info = []
        random_negative_node_id = []
        batch_random_num_node = []
        batch_random_composition_info = []
        batch_random_original_pos = []
        batch_random_negative_node_id = []
        # generate random binary tree for all sentence in training data each epoch
        for tree in self.tree_list:
            random_tree_info = tree.generate_random_tree()
            random_num_node.append(random_tree_info[0])
            random_composition_info.append(
                torch.tensor(
                    random_tree_info[1],
                    dtype=torch.long,
                    device=self.device))
            random_original_position.append(
                torch.tensor(
                    random_tree_info[2],
                    dtype=torch.long,
                    device=self.device))
            random_negative_node_id.append(
                torch.tensor(
                    random_tree_info[3],
                    dtype=torch.long,
                    device=self.device))
        if batch_size is None:
            batch_tree_id_list = list(range(num_tree))
            batch_num_node.append(
                list(itemgetter(*batch_tree_id_list)(self.num_node)))
            batch_sentence_list.append(list(itemgetter(*batch_tree_id_list)(self.sentence_list)))
            batch_label_list.append(list(itemgetter(
                *batch_tree_id_list)(self.label_list)))
            batch_original_position.append(
                list(
                    itemgetter(
                        *
                        batch_tree_id_list)(
                        self.original_position)))
            batch_composition_info.append(list(itemgetter(
                *batch_tree_id_list)(self.composition_info)))
            batch_word_split.append(list(itemgetter(
                *batch_tree_id_list)(self.word_split)))
            batch_random_num_node.append(list(itemgetter(*batch_tree_id_list)(random_num_node)))
            batch_random_composition_info.append(
                list(itemgetter(*batch_tree_id_list)(random_composition_info)))
            batch_random_original_pos.append(
                list(itemgetter(*batch_tree_id_list)(random_original_position)))
            batch_random_negative_node_id.append(
                list(itemgetter(*batch_tree_id_list)(random_negative_node_id)))
        else:
            # shuffle the tree_id in tree_list
            shuffled_tree_id = self.make_shuffled_tree_id()
            for idx in range(0, num_tree - batch_size, batch_size):
                batch_tree_id_list = shuffled_tree_id[idx:idx + batch_size]
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
                batch_original_position.append(
                    list(
                        itemgetter(
                            *
                            batch_tree_id_list)(
                            self.original_position)))
                batch_composition_info.append(list(itemgetter(
                    *batch_tree_id_list)(self.composition_info)))
                batch_word_split.append(list(itemgetter(
                    *batch_tree_id_list)(self.word_split)))
                batch_random_num_node.append(list(itemgetter(*batch_tree_id_list)(random_num_node)))
                batch_random_composition_info.append(
                    list(itemgetter(*batch_tree_id_list)(random_composition_info)))
                batch_random_original_pos.append(
                    list(itemgetter(*batch_tree_id_list)(random_original_position)))
                batch_random_negative_node_id.append(
                    list(itemgetter(*batch_tree_id_list)(random_negative_node_id)))
            # the part cannot devided by batch_size
            batch_num_node.append(list(itemgetter(
                *shuffled_tree_id[idx + batch_size:])(self.num_node)))
            batch_sentence_list.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(self.sentence_list)))
            batch_label_list.append(list(itemgetter(
                *shuffled_tree_id[idx + batch_size:])(self.label_list)))
            batch_original_position.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(self.original_position)))
            batch_composition_info.append(list(itemgetter(
                *shuffled_tree_id[idx + batch_size:])(self.composition_info)))
            batch_word_split.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(self.word_split)))
            batch_random_num_node.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(random_num_node)))
            batch_random_composition_info.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(random_composition_info)))
            batch_random_original_pos.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(random_original_position)))
            batch_random_negative_node_id.append(
                list(itemgetter(*shuffled_tree_id[idx + batch_size:])(random_negative_node_id)))

        for idx in range(len(batch_composition_info)):
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

        for idx in range(len(batch_random_composition_info)):
            composition_list = batch_random_composition_info[idx]
            # set mask for composition info in each batch
            max_num_composition = max([len(i) for i in composition_list])
            # make dummy compoisition info to fill blank in batch
            dummy_compositin_info = [
                torch.ones(
                    max_num_composition - len(i),
                    4,
                    dtype=torch.long,
                    device=self.device) * -1 for i in composition_list]
            batch_random_composition_info[idx] = torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(composition_list, dummy_compositin_info)])
        # return zipped batch information, when training, extract each batch from zip itteration
        batch_list = list(zip(
            batch_num_node,
            batch_sentence_list,
            batch_original_position,
            batch_composition_info,
            batch_label_list,
            batch_word_split,
            batch_random_num_node,
            batch_random_composition_info,
            batch_random_original_pos,
            batch_random_negative_node_id))
        return batch_list

    def set_vector(self, holccg: HolCCG) -> None:
        """set the vector for all node in tree list

        Parameters
        ----------
        holccg
            HolCCG model
        """
        with tqdm(total=len(self.tree_list)) as pbar:
            pbar.set_description("setting vector...")
            for tree in self.tree_list:
                sentence = [" ".join(tree.sentence)]
                word_split = [tree.word_split]
                vector_list, _ = holccg.encode(sentence, word_split=word_split)
                vector_list = vector_list[0]
                for pos in tree.original_position:
                    node_id = pos[0]
                    original_position = pos[1]
                    node = tree.node_list[node_id]
                    node.vector = torch.squeeze(vector_list[original_position])
                for composition_info in tree.composition_info:
                    parent_node = tree.node_list[composition_info[1]]
                    left_node = tree.node_list[composition_info[2]]
                    right_node = tree.node_list[composition_info[3]]
                    if holccg.composition == 'corr':
                        parent_node.vector = circular_correlation(
                            left_node.vector, right_node.vector, holccg.vector_norm)
                    elif holccg.composition == 'conv':
                        parent_node.vector = circular_convolution(
                            left_node.vector, right_node.vector, holccg.vector_norm)
                pbar.update(1)

    def convert_to_binary(self, type: str) -> None:
        """convert tree to binary tree

        Parameters
        ----------
        type
            type of binary tree
        """
        for tree in self.tree_list:
            root_node = tree.node_list[-1]
            root_node.parent_node = None
            for node in reversed(tree.node_list):
                if not node.is_leaf:
                    if node.num_child == 1:
                        child_node = tree.node_list[node.child_node_id]
                        if node.parent_node is None:
                            child_node.parent_node = node
                        else:
                            child_node.parent_node = node.parent_node
                        child_node.category += '-->' + node.category
                    elif node.num_child == 2:
                        left_child_node = tree.node_list[node.left_child_node_id]
                        right_child_node = tree.node_list[node.right_child_node_id]
                        left_child_node.parent_node = node
                        right_child_node.parent_node = node
            binary_node_list = []
            node_id = 0
            for node in tree.node_list:
                if node.is_leaf or node.num_child == 2:
                    node.self_id = node_id
                    node.prime_category = node.category.split('-->')[0]
                    if node.parent_node is not None:
                        parent_node = node.parent_node
                        if parent_node.start_idx == node.start_idx:
                            parent_node.left_child_node_id = node.self_id
                        elif parent_node.end_idx == node.end_idx:
                            parent_node.right_child_node_id = node.self_id
                    binary_node_list.append(node)
                    node_id += 1
            tree.node_list = binary_node_list
        if type == 'train':
            self.set_vocab_and_head()
        self.set_category_id()

    def count_rule(self) -> Counter:
        """count rule in tree list

        Returns
        -------
        rule_counter
            rule counter
        """
        rule_counter = Counter()
        for tree in self.tree_list:
            for node in tree.node_list:
                if not node.is_leaf:
                    left_child_node = tree.node_list[node.left_child_node_id]
                    right_child_node = tree.node_list[node.right_child_node_id]
                    left = left_child_node.category.split('-->')[-1]
                    right = right_child_node.category.split('-->')[-1]
                    parent = node.category
                    rule_counter[(left, right, parent)] += 1
        return rule_counter
