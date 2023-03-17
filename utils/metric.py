from dataclasses import dataclass
from typing import List, Set


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


@dataclass
class Score:
    num_gold: int = 0
    num_pred: int = 0
    num_match: int = 0
    num_match_cls: int = 0

    def add_gold(self, num):
        self.num_gold += num

    def add_pred(self, num):
        self.num_pred += num

    def add_match(self, num):
        self.num_match += num

    def add_match_cls(self, num):
        self.num_match_cls += num

    def compute_f1(self, num_match):
        precision = safe_div(num_match, self.num_pred)
        recall = safe_div(num_match, self.num_gold)
        f1 = safe_div(2 * precision * recall, precision + recall)
        return {'precision': precision * 100, 'recall': recall * 100, 'f1': f1 * 100}

    def compute(self):
        id_result = self.compute_f1(self.num_match)
        cls_result = self.compute_f1(self.num_match_cls)
        return {'id': id_result, 'cls': cls_result}


def compute_matched(predictions: Set, golds: Set):
    num_match_cls = len(predictions & golds)
    pred_pos = {p[:-1] for p in predictions}
    gold_pos = {g[:-1] for g in golds}
    num_match = len(pred_pos & gold_pos)
    return num_match, num_match_cls


def convert_events(events):
    triggers = set()
    arguments = set()
    for event in events:
        triggers.add((event.trigger.start, event.trigger.end, event.trigger.label))
        for argument in event.arguments:
            arguments.add((argument.start, argument.end, event.trigger.label, argument.label))
    return triggers, arguments


def is_overlap_arg(gold_example):
    arguments = []
    for i, event in enumerate(gold_example.events):
        for j, arg in enumerate(event.arguments):
            arguments.append([arg, i, j])
    overlap_type = None
    for i in range(len(arguments)):
        cur_i, cur_j = arguments[i][1:]
        cur_arg = (arguments[i][0].start, arguments[i][0].end)
        for j in range(i + 1, len(arguments)):
            tar_i, tar_j = arguments[j][1:]
            tar_arg = (arguments[j][0].start, arguments[j][0].end)
            if (cur_arg == tar_arg) and i != j:
                if cur_i == tar_i:
                    overlap_type = 'single'
                elif cur_i != tar_i:
                    overlap_type = 'cross'
    return overlap_type


def score(predictions, golds):
    trigger_scorer = Score()
    argument_scorer = Score()
    single_overlap_scorer = Score()
    cross_overlap_scorer = Score()
    for prediction, gold in zip(predictions, golds):
        pred_triggers, pred_arguments = convert_events(prediction.events)
        gold_triggers, gold_arguments = convert_events(gold.events)

        trigger_scorer.add_pred(len(pred_triggers))
        trigger_scorer.add_gold(len(gold_triggers))
        tri_match, tri_match_cls = compute_matched(pred_triggers, gold_triggers)
        trigger_scorer.add_match(tri_match)
        trigger_scorer.add_match_cls(tri_match_cls)

        argument_scorer.add_pred(len(pred_arguments))
        argument_scorer.add_gold(len(gold_arguments))
        arg_match, arg_match_cls = compute_matched(pred_arguments, gold_arguments)
        argument_scorer.add_match(arg_match)
        argument_scorer.add_match_cls(arg_match_cls)

        if is_overlap_arg(gold) == 'single':
            single_overlap_scorer.add_pred(len(pred_arguments))
            single_overlap_scorer.add_gold(len(gold_arguments))
            arg_match, arg_match_cls = compute_matched(pred_arguments, gold_arguments)
            single_overlap_scorer.add_match(arg_match)
            single_overlap_scorer.add_match_cls(arg_match_cls)
        if is_overlap_arg(gold) == 'cross':
            cross_overlap_scorer.add_pred(len(pred_arguments))
            cross_overlap_scorer.add_gold(len(gold_arguments))
            arg_match, arg_match_cls = compute_matched(pred_arguments, gold_arguments)
            cross_overlap_scorer.add_match(arg_match)
            cross_overlap_scorer.add_match_cls(arg_match_cls)

    results = {
        'trigger-id': trigger_scorer.compute()['id'],
        'trigger-cls': trigger_scorer.compute()['cls'],
        'argument-id': argument_scorer.compute()['id'],
        'argument-cls': argument_scorer.compute()['cls'],
        # 'cross-overlap': cross_overlap_scorer.compute()['cls'],
        # 'single-overlap': single_overlap_scorer.compute()['cls']
    }

    metric = {}
    for k, v in results.items():
        for k2, v2 in v.items():
            metric[f'{k}-{k2}'] = v2
    return metric
