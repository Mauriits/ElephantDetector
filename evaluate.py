import cv2
import numpy as np
import nms
import math


def evaluate_detection(bounding_boxes, window_amount, correct_bboxes=None):

    iou_threshold = 0.7
    true_positives = 0

    if correct_bboxes is not None:
        for correct_bbox in correct_bboxes:
            for bounding_box in bounding_boxes:
                if nms.intersection_over_union(bounding_box, correct_bbox) > iou_threshold:
                    true_positives += 1
                    break

        false_negatives = len(correct_bboxes) - true_positives
    else:
        true_positives = 0
        false_negatives = 0

    false_positives = len(bounding_boxes) - true_positives

    if correct_bboxes is not None:
        true_negatives = window_amount - len(correct_bboxes) - false_positives
    else:
        true_negatives = window_amount - false_positives

    return true_positives, false_negatives, false_positives, true_negatives


def calc_fscore(guessed_labels, correct_labels):
    precision = calc_precision(guessed_labels, correct_labels)
    recall = calc_recall(guessed_labels, correct_labels)

    return 2 * precision * recall / (precision + recall)


def calc_precision(guessed_labels, correct_labels):

    guesses = len(guessed_labels)
    correct = 0
    for i, score in enumerate(guessed_labels):
        if np.sign(score[0]) == correct_labels[i]:
            correct += 1

    return correct / guesses


def calc_recall(guessed_labels, correct_labels):

    trues = 0
    found = 0
    for i, score in enumerate(guessed_labels):
        if correct_labels[i] == -1:
            continue

        trues += 1
        if np.sign(score[0]) == 1:
            found += 1

    return found / trues


def calc_prec_recall(threshold_tp, threshold_fn, threshold_fp):
    if (threshold_tp + threshold_fp == 0):
        precision = 1
    else:
        precision = threshold_tp / (threshold_tp + threshold_fp)

    if threshold_tp + threshold_fn == 0:
        recall = 1
    else:
        recall = threshold_tp / (threshold_tp + threshold_fn)

    return precision, recall


def calc_fmeasure(threshold_tp, threshold_fn, threshold_fp):
    precision, recall = calc_prec_recall(threshold_tp, threshold_fn, threshold_fp)

    if recall == 0 or precision == 0:
        fmeasure = 0
    else:
        fmeasure = math.ceil((2 * precision * recall / (precision + recall)) * 100) / 100

    return fmeasure


def show_pr_graph(threshold_tp, threshold_fn, threshold_fp, highlight=None):
    canvas = np.ones((550, 680, 3), np.uint8) * 255

    size = 400
    x = 220
    y = 50

    # Axes text
    cv2.putText(canvas, "Recall", (390, 515), 2, 1, (0, 0, 0), 0, cv2.LINE_AA)
    cv2.putText(canvas, "Precision", (20, 260), 2, 1, (0, 0, 0), 0, cv2.LINE_AA)

    numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cv2.putText(canvas, str(0), (x - 5, y + size + 18), 1, 1, (0, 0, 0), 0, cv2.LINE_AA)
    cv2.putText(canvas, str(0), (x - 19, y + size + 4), 1, 1, (0, 0, 0), 0, cv2.LINE_AA)

    offset = 0
    for n in numbers:

        # Lines and numbers
        cv2.putText(canvas, str(n), (x - 32, y + size - offset - 34), 1, 1, (0, 0, 0), 0, cv2.LINE_AA)
        cv2.putText(canvas, str(n), (x + 28 + offset, y + size + 18), 1, 1, (0, 0, 0), 0, cv2.LINE_AA)
        offset += int(size / 10)
        cv2.line(canvas, (x + offset, y), (x + offset, y + size), (140, 140, 140))
        cv2.line(canvas, (x, y + offset), (x + size, y + offset), (140, 140, 140))


    cv2.rectangle(canvas, (x, y), (x + size, y + size), (0, 0, 0), 1)

    # Draw graph
    for i in range(len(threshold_tp)):

        precision, recall = calc_prec_recall(threshold_tp[i], threshold_fn[i], threshold_fp[i])

        color = (200, 0, 0)
        t=2
        if i == highlight:
            color = (0, 0, 255)
            t = 4
        cv2.circle(canvas, (x + int(recall * 400), y + size - int(precision * 400)), 4, color, t)

    # F1 score
    #cv2.putText(canvas, "Average: " , (350, 25), 1, 1, (0, 0, 0), 0, cv2.LINE_AA)

    cv2.imshow("Precision-recall graph", canvas)