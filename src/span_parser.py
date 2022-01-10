import sys
from utils import load, Condition_Setter
import time
import torch
from utils import circular_correlation
from grammar import Combinator


class Category:
    def __init__(
            self,
            cell_id,
            cat,
            type,
            vector,
            total_ll,
            label_ll,
            span_ll=None,
            num_child=None,
            left_child=None,
            right_child=None,
            head=None,
            is_leaf=False,
            word=None):
        self.cell_id = cell_id
        self.cat = cat
        self.type = type
        self.vector = vector
        self.total_ll = total_ll
        self.label_ll = label_ll
        self.span_ll = span_ll
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
        if category.cat in self.best_category_id:
            best_category = self.category_list[self.best_category_id[category.cat]]
            # only when the new category has higher probability than existing one, replace it
            if category.total_ll > best_category.total_ll:
                self.best_category_id[category.cat] = len(self.category_list)
                self.category_list.append(category)
        else:
            self.best_category_id[category.cat] = len(self.category_list)
            self.category_list.append(category)
            return self.best_category_id[category.cat]


class Parser:
    def __init__(
            self,
            tree_net,
            combinator,
            word_category_vocab,
            phrase_category_vocab,
            stag_threshold,
            phrase_threshold,
            span_threshold,
            max_parse_time=60):
        self.tree_net = tree_net
        self.word_ff = tree_net.word_ff
        self.phrase_ff = tree_net.phrase_ff
        self.span_ff = tree_net.span_ff
        self.combinator = combinator
        self.word_category_vocab = word_category_vocab
        self.phrase_category_vocab = phrase_category_vocab
        self.stag_threshold = stag_threshold
        self.phrase_threshold = phrase_threshold
        self.span_threshold = span_threshold
        self.max_parse_time = max_parse_time

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
        word_split = self.tree_net.set_word_split(converted_sentence)
        word_vectors, _ = self.tree_net.encode([" ".join(converted_sentence)], [word_split])
        word_vectors = word_vectors[0]
        word_probs_list = torch.softmax(self.word_ff(word_vectors), dim=-1)
        word_predict_cats = torch.argsort(word_probs_list, descending=True)
        word_predict_cats = word_predict_cats[word_predict_cats != 0].view(
            word_probs_list.shape[0], -1)

        chart = {}

        for idx in range(len(converted_sentence)):
            word = sentence[idx]
            vector = word_vectors[idx]
            word_probs = word_probs_list[idx]
            top_cat_id = word_predict_cats[idx, 0]
            top_category = Category(
                cell_id=(idx, idx + 1),
                cat=self.word_category_vocab.itos[top_cat_id],
                type='stag',
                vector=vector,
                total_ll=torch.log(word_probs[top_cat_id]),
                label_ll=torch.log(word_probs[top_cat_id]),
                is_leaf=True,
                word=word)
            chart[(idx, idx + 1)] = Cell(word)
            chart[(idx, idx + 1)].add_category(top_category)

            for cat_id in word_predict_cats[idx, 1:]:
                if word_probs[cat_id] > self.stag_threshold:
                    category = Category(cell_id=(idx,
                                                 idx + 1),
                                        cat=self.word_category_vocab.itos[cat_id],
                                        type='stag',
                                        vector=vector,
                                        total_ll=torch.log(word_probs[cat_id]),
                                        label_ll=torch.log(word_probs[cat_id]),
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
                    possible_cats_info = self.combinator.unary_rule.get(child_cat.cat)
                    if possible_cats_info is None:
                        continue
                    else:
                        span_prob = torch.sigmoid(self.span_ff(child_cat.vector))
                        if span_prob > self.span_threshold:
                            span_ll = torch.log(span_prob)
                            phrase_probs = torch.softmax(self.phrase_ff(child_cat.vector), dim=-1)
                            for parent_cat_info in possible_cats_info:
                                parent_cat = parent_cat_info[0]
                                parent_cat_id = self.phrase_category_vocab[parent_cat]
                                # when target category is not <unk>
                                if parent_cat_id != 0:
                                    parent_cat_type = parent_cat_info[1]
                                    label_prob = phrase_probs[parent_cat_id]
                                    if label_prob > self.phrase_threshold:
                                        label_ll = torch.log(label_prob)
                                        total_ll = label_ll + span_ll + child_cat.total_ll
                                        parent_category = Category(
                                            cell_id=(idx, idx + 1),
                                            cat=parent_cat,
                                            type=parent_cat_type,
                                            vector=child_cat.vector,
                                            total_ll=total_ll,
                                            label_ll=label_ll,
                                            span_ll=span_ll,
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
        start = time.time()
        chart = self.initialize_chart(sentence)
        n = len(chart)
        for length in range(2, n + 1):
            for left in range(n - length + 1):
                right = left + length
                chart[(left, right)] = Cell(' '.join(sentence.split()[left:right]))
                # when time is not over
                if time.time() - start < self.max_parse_time:
                    for split in range(left + 1, right):
                        for left_cat_id in chart[(left, split)].best_category_id.values():
                            left_cat = chart[(left, split)].category_list[left_cat_id]
                            for right_cat_id in chart[(split, right)].best_category_id.values():
                                right_cat = chart[(split, right)].category_list[right_cat_id]
                                # list of gramatically possible category
                                possible_cats_info = self.combinator.binary_rule.get(
                                    (left_cat.cat, right_cat.cat))
                                if possible_cats_info is None:
                                    continue
                                else:
                                    # apply filter to possible categories based on Eisner
                                    # constraints
                                    if left_cat.type == 'fc':
                                        filtered_parent_cat_info = []
                                        for parent_cat_info in possible_cats_info:
                                            if parent_cat_info[1] not in ['fa', 'fc']:
                                                filtered_parent_cat_info.append(parent_cat_info)
                                        possible_cats_info = filtered_parent_cat_info
                                    if right_cat.type == 'bc':
                                        filtered_parent_cat_info = []
                                        for parent_cat_info in possible_cats_info:
                                            if parent_cat_info[1] not in ['ba', 'bc']:
                                                filtered_parent_cat_info.append(parent_cat_info)
                                        possible_cats_info = filtered_parent_cat_info
                                    composed_vector = circular_correlation(
                                        left_cat.vector, right_cat.vector)
                                    span_prob = torch.sigmoid(self.span_ff(composed_vector))
                                    if span_prob > self.span_threshold:
                                        span_ll = torch.log(span_prob)
                                        phrase_probs = torch.softmax(
                                            self.phrase_ff(composed_vector), dim=-1)
                                        for parent_cat_info in possible_cats_info:
                                            parent_cat = parent_cat_info[0]
                                            parent_cat_id = self.phrase_category_vocab[parent_cat]
                                            # when category is not None
                                            if parent_cat_id != 0:
                                                parent_cat_type = parent_cat_info[1]
                                                label_prob = phrase_probs[parent_cat_id]
                                                if label_prob > self.phrase_threshold:
                                                    label_ll = torch.log(label_prob)
                                                    total_ll = label_ll + span_ll + left_cat.total_ll + right_cat.total_ll
                                                    head = self.combinator.head_info[(
                                                        left_cat.cat, right_cat.cat, parent_cat)]
                                                    parent_category = Category(
                                                        cell_id=(left, right),
                                                        cat=parent_cat,
                                                        type=parent_cat_type,
                                                        vector=composed_vector,
                                                        total_ll=total_ll,
                                                        label_ll=label_ll,
                                                        span_ll=span_ll,
                                                        num_child=2,
                                                        left_child=left_cat,
                                                        right_child=right_cat,
                                                        head=head)
                                                    chart[(left, right)].add_category(
                                                        parent_category)

                    waiting_cat_id = list(chart[(left, right)].best_category_id.values())
                    while True:
                        if waiting_cat_id == []:
                            break
                        else:
                            child_cat_id = waiting_cat_id.pop(0)
                            child_cat = chart[(left, right)].category_list[child_cat_id]
                            possible_cats_info = self.combinator.unary_rule.get(child_cat.cat)
                            if possible_cats_info is None:
                                continue
                            else:
                                span_prob = torch.sigmoid(self.span_ff(child_cat.vector))
                                if span_prob > self.span_threshold:
                                    span_ll = torch.log(span_prob)
                                    phrase_probs = torch.softmax(
                                        self.phrase_ff(child_cat.vector), dim=-1)
                                    for parent_cat_info in possible_cats_info:
                                        parent_cat = parent_cat_info[0]
                                        parent_cat_id = self.phrase_category_vocab[parent_cat]
                                        # when category is not <unk>
                                        if parent_cat_id != 0:
                                            parent_cat_type = parent_cat_info[1]
                                            label_prob = phrase_probs[parent_cat_id]
                                            if label_prob > self.phrase_threshold:
                                                label_ll = torch.log(label_prob)
                                                total_ll = label_ll + span_ll + child_cat.total_ll
                                                parent_category = Category(
                                                    cell_id=(left, right),
                                                    cat=parent_cat,
                                                    type=parent_cat_type,
                                                    vector=child_cat.vector,
                                                    total_ll=total_ll,
                                                    label_ll=label_ll,
                                                    span_ll=span_ll,
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

    def skimmer(self, chart):
        len_sentence = list(chart.keys())[-1][1]
        found_span_id = []
        if len_sentence == 1:
            found_span_id.append((0, 1))
        else:
            cell_id_list = list(chart.keys())
            scope_list = [(0, len_sentence)]
            while scope_list != []:
                scope = scope_list.pop(0)
                for cell_id in cell_id_list:
                    if scope[0] <= cell_id[0] and cell_id[1] <= scope[1]:
                        cell = chart[cell_id]
                        max_span_length = 0
                        if len(cell.best_category_id) != 0:
                            span_length = cell_id[1] - cell_id[0]
                            if span_length > max_span_length:
                                max_span_length = span_length
                                max_span_id = cell_id
                found_span_id.append(max_span_id)
                cell_id_list.remove(max_span_id)
                left_scope = (scope[0], max_span_id[0])
                right_scope = (max_span_id[1], scope[1])
                if left_scope[1] - left_scope[0] > 1:
                    scope_list.append(left_scope)
                elif left_scope[1] - left_scope[0] == 1:
                    found_span_id.append(left_scope)
                if right_scope[1] - right_scope[0] > 1:
                    scope_list.append(right_scope)
                elif right_scope[1] - right_scope[0] == 1:
                    found_span_id.append(right_scope)
        autos = []
        auto_scopes = []
        for span_id in found_span_id:
            auto = self.decode(chart[span_id])
            autos.append(auto)
            auto_scopes.append(span_id)

        sorted_autos = []
        sorted_auto_scopes = []
        target = 0
        while True:
            for i in range(len(auto_scopes)):
                scope = auto_scopes[i]
                start = scope[0]
                end = scope[1]
                if start == target:
                    sorted_autos.append(autos[i])
                    sorted_auto_scopes.append(auto_scopes[i])
                    target = end
                    break
            if target == len_sentence:
                break
        return sorted_autos, sorted_auto_scopes

    def decode(self, root_cell):
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

        root_cat = root_cell.category_list[0]
        for cat in root_cell.category_list[1:]:
            if cat.total_ll > root_cat.total_ll:
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


def main():
    condition = Condition_Setter(set_embedding_type=False)

    args = sys.argv
    # args = [
    #     '',
    #     'roberta-large_phrase_span_2022-01-08_17:57:10.pth',
    #     'dev',
    #     '0.1',
    #     '0.01',
    #     '0.05',
    #     '10']

    model = args[1]
    dev_test = args[2]
    stag_threhold = float(args[3])
    phrase_threshold = float(args[4])
    span_threshold = float(args[5])
    min_freq = int(args[6])

    if dev_test == 'dev':
        path_to_sentence_list = "CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw"
    else:
        path_to_sentence_list = "CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tree_net = torch.load(condition.path_to_model + model,
                          map_location=device)
    tree_net.device = device
    tree_net.eval()

    word_category_vocab = load(condition.path_to_word_category_vocab)
    phrase_category_vocab = load(condition.path_to_phrase_category_vocab)

    combinator = Combinator(condition.PATH_TO_DIR + "span_parsing/GRAMMAR/", min_freq=min_freq)

    parser = Parser(
        tree_net,
        combinator=combinator,
        word_category_vocab=word_category_vocab,
        phrase_category_vocab=phrase_category_vocab,
        stag_threshold=stag_threhold,
        phrase_threshold=phrase_threshold,
        span_threshold=span_threshold)

    with open(condition.PATH_TO_DIR + path_to_sentence_list, 'r') as f:
        sentence_list = f.readlines()

    sentence_id = 0
    for sentence in sentence_list:
        sentence_id += 1
        sentence = sentence.rstrip()
        chart = parser.parse(sentence)
        root_cell = list(chart.values())[-1]
        if len(root_cell.best_category_id) == 0:
            autos, scope_list = parser.skimmer(chart)
            n = 0
            for auto, scope in zip(autos, scope_list):
                print(
                    'ID={}.{} PARSER=TEST APPLY_SKIMMER=True SCOPE=({},{})'.format(
                        sentence_id, n, scope[0], scope[1]))
                print(auto)
                n += 1
        else:
            auto = parser.decode(root_cell)
            print('ID={} PARSER=TEST APPLY_SKIMMER=FALSE'.format(sentence_id))
            print(auto)


if __name__ == "__main__":
    main()