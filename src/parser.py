from os import wait
import numpy as np
from utils import load, Condition_Setter
import time
import torch
from utils import single_circular_correlation


class Category:
    def __init__(
            self,
            cell_id,
            cat,
            cat_id,
            vector,
            total_score,
            label_score,
            span_score=None,
            num_child=None,
            left_child=None,
            right_child=None,
            head=None,
            is_leaf=False,
            word=None):
        self.cell_id = cell_id
        self.cat = cat
        self.cat_id = cat_id
        self.vector = vector
        self.total_score = total_score
        self.label_score = label_score
        self.span_score = span_score
        self.num_child = num_child
        self.left_child = left_child
        self.right_child = right_child
        self.head = head
        self.is_leaf = is_leaf
        self.word = word


class Cell:
    def __init__(self, content):
        self.content = content
        self.category_list = []
        self.best_category_id = {}

    def add_category(self, category):
        # when category already exist in the cell
        if category.cat_id in self.best_category_id:
            best_category = self.category_list[self.best_category_id[category.cat_id]]
            # only when the new category has higher score than existing one, replace it
            if category.total_score > best_category.total_score:
                self.best_category_id[category.cat_id] = len(self.category_list)
                self.category_list.append(category)
        else:
            self.best_category_id[category.cat_id] = len(self.category_list)
            self.category_list.append(category)
            return self.best_category_id[category.cat_id]


class Parser:
    def __init__(
            self,
            tree_net,
            binary_rule,
            unary_rule,
            head_info,
            category_vocab,
            word_to_whole,
            whole_to_phrase,
            stag_threshold,
            label_threshold,
            span_threshold):
        self.tokenizer = tree_net.tokenizer
        self.encoder = tree_net.model
        self.word_ff = tree_net.word_ff.to('cpu')
        self.phrase_ff = tree_net.phrase_ff.to('cpu')
        self.span_ff = tree_net.span_ff.to('cpu')
        self.binary_rule = binary_rule
        self.unary_rule = unary_rule
        self.head_info = head_info
        self.category_vocab = category_vocab
        self.word_to_whole = word_to_whole
        self.whole_to_phrase = whole_to_phrase
        self.stag_threshold = stag_threshold
        self.label_threshold = label_threshold
        self.span_threshold = span_threshold

    def initialize_chart(self, sentence):
        sentence = sentence.split()
        converted_sentence = []
        for i in range(len(sentence)):
            content = sentence[i]
            if content == "-LRB-":
                content = "("
            elif content == "-LCB-":
                content = "{"
            elif content == "-RRB-":
                content = ")"
            elif content == "-RCB-":
                content = "}"
            if r"\/" in content:
                content = content.replace(r"\/", "/")
            converted_sentence.append(content)
        tokens = self.tokenizer.tokenize(" ".join(converted_sentence))
        tokenized_pos = 0
        word_split = []
        for original_pos in range(len(converted_sentence)):
            word = converted_sentence[original_pos]
            length = 1
            while True:
                temp = self.tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length]).replace(" ", "")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        input = self.tokenizer(
            " ".join(converted_sentence),
            return_tensors='pt').to(self.encoder.device)
        output = self.encoder(**input).last_hidden_state[0, 1:-1].to('cpu')
        temp = []
        for start_idx, end_idx in word_split:
            temp.append(torch.mean(output[start_idx:end_idx], dim=0))
        word_vectors = torch.stack(temp)
        word_scores = self.word_ff(word_vectors)
        word_prob = torch.softmax(word_scores, dim=-1)
        word_predict_cats = torch.argsort(word_prob, descending=True)
        word_predict_cats = word_predict_cats[word_predict_cats != 0].view(word_prob.shape[0], -1)

        chart = {}

        for idx in range(len(converted_sentence)):
            word = sentence[idx]
            vector = word_vectors[idx]
            score = word_scores[idx]
            prob = word_prob[idx]
            top_cat_id = word_predict_cats[idx, 0]
            top_category = Category(
                (idx, idx + 1),
                self.category_vocab.itos[self.word_to_whole[top_cat_id]],
                self.word_to_whole[top_cat_id],
                vector,
                total_score=score[top_cat_id],
                label_score=score[top_cat_id],
                is_leaf=True,
                word=word)
            chart[(idx, idx + 1)] = Cell(word)
            chart[(idx, idx + 1)].add_category(top_category)

            for cat_id in word_predict_cats[idx, 1:]:
                if prob[cat_id] > self.stag_threshold:
                    category = Category((idx,
                                         idx + 1),
                                        self.category_vocab.itos[self.word_to_whole[cat_id]],
                                        self.word_to_whole[cat_id],
                                        vector,
                                        score[cat_id],
                                        score[cat_id],
                                        is_leaf=True,
                                        word=word)
                    chart[(idx, idx + 1)].add_category(category)
                else:
                    break

            waiting_cat_id = list(chart[(idx, idx + 1)].best_category_id.values())
            while True:
                if waiting_cat_id == []:
                    break
                else:
                    child_cat_id = waiting_cat_id.pop(0)
                    child_cat = chart[(idx, idx + 1)].category_list[child_cat_id]
                    possible_cat_id = self.unary_rule.get(child_cat.cat_id)
                    if possible_cat_id is None:
                        continue
                    else:
                        span_score = self.span_ff(child_cat.vector)
                        span_prob = torch.sigmoid(span_score)
                        if span_prob > self.span_threshold:
                            phrase_scores = self.phrase_ff(child_cat.vector)
                            phrase_probs = torch.softmax(phrase_scores, dim=-1)
                            for parent_cat_id in possible_cat_id:
                                cat = self.category_vocab.itos[parent_cat_id]
                                label_score = phrase_scores[self.whole_to_phrase[parent_cat_id]]
                                label_prob = phrase_probs[self.whole_to_phrase[parent_cat_id]]
                                if label_prob > self.label_threshold:
                                    total_score = label_score + span_score + child_cat.total_score
                                    parent_category = Category(
                                        (idx, idx + 1),
                                        cat,
                                        parent_cat_id,
                                        child_cat.vector,
                                        total_score=total_score,
                                        label_score=label_score,
                                        span_score=span_score,
                                        num_child=1,
                                        left_child=child_cat,
                                        head=0)
                                    new_cat_id = chart[(idx, idx + 1)].add_category(
                                        parent_category)
                                    if new_cat_id is None:
                                        continue
                                    else:
                                        waiting_cat_id.append(new_cat_id)
        return chart

    @torch.no_grad()
    def parse(self, sentence):
        chart = self.initialize_chart(sentence)
        n = len(chart)
        for length in range(2, n + 1):
            for left in range(n - length + 1):
                right = left + length
                chart[(left, right)] = Cell(' '.join(sentence.split()[left:right]))
                for split in range(left + 1, right):
                    for left_cat_id in chart[(left, split)].best_category_id.values():
                        left_cat = chart[(left, split)].category_list[left_cat_id]
                        for right_cat_id in chart[(split, right)].best_category_id.values():
                            right_cat = chart[(split, right)].category_list[right_cat_id]
                            # list of gramatically possible category
                            possible_cat_id = self.binary_rule.get(
                                (left_cat.cat_id, right_cat.cat_id))
                            if possible_cat_id is None:
                                continue
                            else:
                                composed_vector = single_circular_correlation(
                                    left_cat.vector, right_cat.vector)
                                span_score = self.span_ff(composed_vector)
                                span_prob = torch.sigmoid(span_score)
                                # print(chart[(left, split)].content,
                                #       chart[(split, right)].content, span_score)
                                if span_prob > self.span_threshold:
                                    phrase_scores = self.phrase_ff(composed_vector)
                                    phrase_probs = torch.softmax(phrase_scores, dim=-1)
                                    for parent_cat_id in possible_cat_id:
                                        cat = self.category_vocab.itos[parent_cat_id]
                                        label_score = phrase_scores[self.whole_to_phrase[parent_cat_id]]
                                        label_prob = phrase_probs[self.whole_to_phrase[parent_cat_id]]
                                        if label_prob > self.label_threshold:
                                            total_score = label_score + span_score + left_cat.total_score + right_cat.total_score
                                            head = self.head_info[(
                                                left_cat.cat_id, right_cat.cat_id, parent_cat_id)]
                                            parent_category = Category(
                                                (left, right),
                                                cat,
                                                parent_cat_id,
                                                composed_vector,
                                                total_score=total_score,
                                                label_score=label_score,
                                                span_score=span_score,
                                                num_child=2,
                                                left_child=left_cat,
                                                right_child=right_cat,
                                                head=head)
                                            chart[(left, right)].add_category(parent_category)

                waiting_cat_id = list(chart[(left, right)].best_category_id.values())
                while True:
                    if waiting_cat_id == []:
                        break
                    else:
                        child_cat_id = waiting_cat_id.pop(0)
                        child_cat = chart[(left, right)].category_list[child_cat_id]
                        possible_cat_id = self.unary_rule.get(child_cat.cat_id)
                        if possible_cat_id is None:
                            continue
                        else:
                            span_score = self.span_ff(child_cat.vector)
                            span_prob = torch.sigmoid(span_score)
                            if span_prob > self.span_threshold:
                                phrase_scores = self.phrase_ff(child_cat.vector)
                                phrase_probs = torch.softmax(phrase_scores, dim=-1)
                                for parent_cat_id in possible_cat_id:
                                    cat = self.category_vocab.itos[parent_cat_id]
                                    label_score = phrase_scores[self.whole_to_phrase[parent_cat_id]]
                                    label_prob = phrase_probs[self.whole_to_phrase[parent_cat_id]]
                                    if label_prob > self.label_threshold:
                                        total_score = label_score + span_score + child_cat.total_score
                                        parent_category = Category(
                                            (left, right),
                                            cat,
                                            parent_cat_id,
                                            child_cat.vector,
                                            total_score=total_score,
                                            label_score=label_score,
                                            span_score=span_score,
                                            num_child=1,
                                            left_child=child_cat,
                                            head=0)
                                        new_cat_id = chart[(left, right)].add_category(
                                            parent_category)
                                        if new_cat_id is None:
                                            continue
                                        else:
                                            waiting_cat_id.append(new_cat_id)
        return chart

    def decode(self, chart):

        def next(waiting_cats):
            cat = waiting_cats.pop(0)
            if cat.is_leaf:
                cat.auto.append('(<L')
                cat.auto.append(cat.cat)
                cat.auto.append('POS')
                cat.auto.append('POS')
                cat.auto.append(cat.word)
                cat.auto.append(cat.cat + '>)')
            elif cat.num_child == 1:
                child_cat = cat.left_child
                child_cat.auto = []
                cat.auto.append('(<T')
                cat.auto.append(cat.cat)
                cat.auto.append('0')
                cat.auto.append('1>')
                cat.auto.append(child_cat.auto)
                cat.auto.append(')')
                waiting_cats.append(child_cat)
            elif cat.num_child == 2:
                left_child_cat = cat.left_child
                right_child_cat = cat.right_child
                left_child_cat.auto = []
                right_child_cat.auto = []
                cat.auto.append('(<T')
                cat.auto.append(cat.cat)
                cat.auto.append(str(cat.head))
                cat.auto.append('2>')
                cat.auto.append(left_child_cat.auto)
                cat.auto.append(right_child_cat.auto)
                cat.auto.append(')')
                waiting_cats.append(left_child_cat)
                waiting_cats.append(right_child_cat)
            return waiting_cats

        def flatten(auto):
            for i in auto:
                if isinstance(i, list):
                    yield from flatten(i)
                else:
                    yield i

        root_cell = list(chart.values())[-1]
        # when fail to parse
        if len(root_cell.best_category_id) == 0:
            # self.skimmer(chart)
            return None
        # when success to parse
        else:
            max_score = -1e+6
            for cat_id in root_cell.best_category_id.values():
                cat = root_cell.category_list[cat_id]
                if cat.total_score > max_score:
                    max_score = cat.total_score
                    root_cat = cat
            root_cat.auto = []
            waiting_cats = [root_cat]
            while True:
                if len(waiting_cats) == 0:
                    break
                else:
                    waiting_cats = next(waiting_cats)

        auto = ' '.join(list(flatten(root_cat.auto)))
        return auto

    def skimmer(self, chart):
        return chart


def extract_rule(path_to_grammar, head_info, category_vocab):
    binary_rule = {}
    unary_rule = {}

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    for rule in data:
        tokens = rule.split()
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]]
            left_cat = category_vocab[tokens[4]]
            right_cat = category_vocab[tokens[5]]
            if (left_cat, right_cat) in binary_rule:
                binary_rule[(left_cat, right_cat)].append(parent_cat)
            else:
                binary_rule[(left_cat, right_cat)] = [parent_cat]
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]]
            child_cat = category_vocab[tokens[4]]
            if child_cat in unary_rule:
                unary_rule[child_cat].append(parent_cat)
            else:
                unary_rule[child_cat] = [parent_cat]
    head_info_temp = head_info
    head_info = {}
    for k, v in head_info_temp.items():
        left_cat_id = category_vocab[k[0]]
        right_cat_id = category_vocab[k[1]]
        parent_cat_id = category_vocab[k[2]]
        head_info[(left_cat_id, right_cat_id, parent_cat_id)] = v
    return binary_rule, unary_rule, head_info


def main():
    condition = Condition_Setter(set_embedding_type=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = "roberta-large_phrase(a).pth"

    tree_net = torch.load(condition.path_to_model + model,
                          map_location=device)
    tree_net.device = device
    tree_net.eval()

    category_vocab = load(condition.path_to_whole_category_vocab)
    word_to_whole = load(condition.path_to_word_to_whole)
    whole_to_phrase = load(condition.path_to_whole_to_phrase)
    head_info = load(condition.path_to_head_info)

    binary_rule, unary_rule, head_info = extract_rule(
        condition.path_to_grammar, head_info, category_vocab)
    parser = Parser(
        tree_net,
        binary_rule,
        unary_rule,
        head_info,
        category_vocab,
        word_to_whole,
        whole_to_phrase,
        stag_threshold=0.075,
        label_threshold=0.001,
        span_threshold=0.1)

    with open(condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw", 'r') as f:
        sentence_list = f.readlines()

    sentence_id = 0
    num_success_sentence = 0
    total_parse_time = 0
    total_decode_time = 0

    for sentence in sentence_list:
        sentence_id += 1
        sentence = sentence.rstrip()
        print('ID={} PARSER=TEST NUMPARSE=1'.format(sentence_id))
        start = time.time()
        chart = parser.parse(sentence)
        time_to_parse = time.time() - start
        total_parse_time += time_to_parse

        start = time.time()
        auto = parser.decode(chart)
        time_to_decode = time.time() - start
        total_decode_time += time_to_decode

        print(auto)

        if auto is not None:
            num_success_sentence += 1
        else:
            break
            # print('faile\n')
    # print("average parse time:{}".format(total_parse_time / len(sentence_list)))
    # print("average decode time:{}".format(total_decode_time / len(sentence_list)))


if __name__ == "__main__":
    main()
