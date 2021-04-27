from collections import Counter
import string
import re
import argparse
import json
import random
import numpy as np
import math
import pdb

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return len(set(normalize_answer(prediction).split()).intersection(set(normalize_answer(ground_truth).split())))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    score = metric_fn(prediction, ground_truths)
    return score


def evaluate_nq(dataset, predictions, total_eval_loss, total_words):
    f1 = exact_match = total = 0
    original_q_nums = 0
    not_answered = 0
    for data, pred in zip(dataset, predictions):
        ground_truths = data.target
        if pred == None:
            pred = 'BLANK'
        prediction = pred
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        total += 1
    exact_match = 100.0 * exact_match / total_words
    f1 = 100.0 * f1 / total_words

    metrics = {}
    metrics["eval_loss"] = np.mean(total_eval_loss)
    metrics["total_words"] = total_words
    metrics["token_accuracy"] = exact_match
    metrics["f1_score"] = f1
    metrics["valid_ppl"] = math.exp(min(np.sum(total_eval_loss)/total_words, 100))

    return metrics
