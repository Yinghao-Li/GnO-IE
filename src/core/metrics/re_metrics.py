import logging
from ..utils import overlaps

logger = logging.getLogger(__name__)


def get_re_metrics(
    pred_list: list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]],
    gt_list: list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]],
    relation_types: list[str],
    tks_list: list[list[str]],
    ids_list: list[int] = None,
    allow_partial_match: bool = False,
):
    """
    Calculate precision, recall, and F1 score for each relation type.

    Parameters
    ----------
    pred_list : list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]]
        A list of dictionaries of predicted relations,
        where each dictionary contains the predicted relations for a paragraph.
        The keys are relation types, and the values are lists of tuples of entity spans.

    gt_list : list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]]
        A list of dictionaries of ground truth relations,
        where each dictionary contains the ground truth relations for a paragraph.
        The keys are relation types, and the values are lists of tuples of entity spans.

    relation_types : list[str]
        A list of relation types to evaluate.

    tks_list : list[list[str]], optional
        A list of tokenized paragraphs, used to compose the error report.

    ids_list : list[int], optional
        A list of paragraph IDs, used to compose the error report.

    allow_partial_match : bool, optional
        Whether to allow entity span partial matches (default is False).
    """

    # Initialize dictionaries to store counts for each entity type
    tp = {relation_type: 0 for relation_type in relation_types}
    fp = {relation_type: 0 for relation_type in relation_types}
    fn = {relation_type: 0 for relation_type in relation_types}

    tp_report = {relation_type: dict() for relation_type in relation_types}
    fp_report = {relation_type: dict() for relation_type in relation_types}
    fn_report = {relation_type: dict() for relation_type in relation_types}

    for idx, predicted, ground_truth, tks in zip(ids_list, pred_list, gt_list, tks_list):
        for relation_type in relation_types:
            pred_rl_spans = predicted[relation_type]
            gt_rl_spans = ground_truth[relation_type]

            if not pred_rl_spans and not gt_rl_spans:
                continue

            if not pred_rl_spans:
                fn[relation_type] += len(gt_rl_spans)

                fn_report[relation_type][idx] = {
                    "text": " ".join(tks),
                    "gt": [
                        (" ".join(tks[span[0][0] : span[0][1]]), " ".join(tks[span[1][0] : span[1][1]]))
                        for span in gt_rl_spans
                    ],
                }

                continue

            if not gt_rl_spans:
                fp[relation_type] += len(pred_rl_spans)

                fp_report[relation_type][idx] = {
                    "text": " ".join(tks),
                    "pred": [
                        (" ".join(tks[span[0][0] : span[0][1]]), " ".join(tks[span[1][0] : span[1][1]]))
                        for span in pred_rl_spans
                    ],
                }

                continue

            # Calculate true positives, false positives, and false negatives for each entity type
            for pred_spans in pred_rl_spans:
                if pred_spans in gt_rl_spans or (
                    allow_partial_match
                    and any(
                        overlaps(pred_spans[0], gt_spans[0]) and overlaps(pred_spans[1], gt_spans[1])
                        for gt_spans in gt_rl_spans
                    )
                ):
                    tp[relation_type] += 1

                    if tp_report[relation_type].get(idx, None) is None:
                        tp_report[relation_type][idx] = {
                            "text": " ".join(tks),
                            "pred": [
                                (
                                    " ".join(tks[pred_spans[0][0] : pred_spans[0][1]]),
                                    " ".join(tks[pred_spans[1][0] : pred_spans[1][1]]),
                                )
                            ],
                        }
                    else:
                        tp_report[relation_type][idx]["pred"].append(
                            (
                                " ".join(tks[pred_spans[0][0] : pred_spans[0][1]]),
                                " ".join(tks[pred_spans[1][0] : pred_spans[1][1]]),
                            )
                        )

                else:
                    fp[relation_type] += 1

                    if fp_report[relation_type].get(idx, None) is None:
                        fp_report[relation_type][idx] = {
                            "text": " ".join(tks),
                            "pred": [
                                (
                                    " ".join(tks[pred_spans[0][0] : pred_spans[0][1]]),
                                    " ".join(tks[pred_spans[1][0] : pred_spans[1][1]]),
                                )
                            ],
                        }
                    else:
                        fp_report[relation_type][idx]["pred"].append(
                            (
                                " ".join(tks[pred_spans[0][0] : pred_spans[0][1]]),
                                " ".join(tks[pred_spans[1][0] : pred_spans[1][1]]),
                            )
                        )

            for gt_spans in gt_rl_spans:
                if not (
                    gt_spans in pred_rl_spans
                    or (
                        allow_partial_match
                        and any(
                            overlaps(pred_spans[0], gt_spans[0]) and overlaps(pred_spans[1], gt_spans[1])
                            for pred_spans in pred_rl_spans
                        )
                    )
                ):
                    fn[relation_type] += 1

                    if fn_report[relation_type].get(idx, None) is None:
                        fn_report[relation_type][idx] = {
                            "text": " ".join(tks),
                            "gt": [
                                (
                                    " ".join(tks[gt_spans[0][0] : gt_spans[0][1]]),
                                    " ".join(tks[gt_spans[1][0] : gt_spans[1][1]]),
                                )
                            ],
                        }
                    else:
                        fn_report[relation_type][idx]["gt"].append(
                            (
                                " ".join(tks[gt_spans[0][0] : gt_spans[0][1]]),
                                " ".join(tks[gt_spans[1][0] : gt_spans[1][1]]),
                            )
                        )

    # Calculate precision, recall, and F1 score for each entity type
    metrics = {}
    for rl_type in relation_types:
        precision = tp[rl_type] / (tp[rl_type] + fp[rl_type] + 1e-9)
        recall = tp[rl_type] / (tp[rl_type] + fn[rl_type] + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        metrics[rl_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[rl_type],
            "fp": fp[rl_type],
            "fn": fn[rl_type],
        }

    # Calculate micro-averaged metrics
    tp = sum(tp.values())
    fp = sum(fp.values())
    fn = sum(fn.values())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    metrics["micro-average"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    reports = {"tp": tp_report, "fp": fp_report, "fn": fn_report}

    return metrics, reports


def get_re_metrics_relaxed(
    pred_list: list[dict[str, list[list[tuple[tuple[int, int]]]]]],
    gt_list: list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]],
    relation_types: list[str],
    tks_list: list[list[str]],
    ids_list: list[int] = None,
    allow_partial_match: bool = False,
):
    """
    Calculate precision, recall, and F1 score for each relation type.

    Parameters
    ----------
    pred_list : list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]]
        A list of dictionaries of predicted relations,
        where each dictionary contains the predicted relations for a paragraph.
        The keys are relation types, and the values are lists of tuples of entity spans.

    gt_list : list[dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]]
        A list of dictionaries of ground truth relations,
        where each dictionary contains the ground truth relations for a paragraph.
        The keys are relation types, and the values are lists of tuples of entity spans.

    relation_types : list[str]
        A list of relation types to evaluate.

    tks_list : list[list[str]], optional
        A list of tokenized paragraphs, used to compose the error report.

    ids_list : list[int], optional
        A list of paragraph IDs, used to compose the error report.

    allow_partial_match : bool, optional
        Whether to allow entity span partial matches (default is False).
    """

    def write_report(report_dict, relation_type, idx, spans):
        if report_dict[relation_type].get(idx, None) is None:
            report_dict[relation_type][idx] = {
                "text": " ".join(tks),
                "spans": [tuple([" ".join(tks[span[0] : span[1]]) for span in spans])],
            }
        else:
            report_dict[relation_type][idx]["spans"].append(tuple([" ".join(tks[span[0] : span[1]]) for span in spans]))
        return None

    # Initialize dictionaries to store counts for each entity type
    tp = {relation_type: 0 for relation_type in relation_types}
    fp = {relation_type: 0 for relation_type in relation_types}
    fn = {relation_type: 0 for relation_type in relation_types}

    tp_report = {relation_type: dict() for relation_type in relation_types}
    fp_report = {relation_type: dict() for relation_type in relation_types}
    fn_report = {relation_type: dict() for relation_type in relation_types}

    for idx, predicted, ground_truth, tks in zip(ids_list, pred_list, gt_list, tks_list):
        for relation_type in relation_types:
            pred_rl_spans = predicted[relation_type]
            gt_rl_spans = ground_truth[relation_type]

            gt_cn_spans = [span[0] for span in gt_rl_spans]
            gt_pn_spans = [span[1] for span in gt_rl_spans]
            gt_pv_spans = [span[2] for span in gt_rl_spans]

            if not pred_rl_spans and not gt_rl_spans:
                continue

            if not pred_rl_spans:
                fn[relation_type] += len(gt_rl_spans)

                fn_report[relation_type][idx] = {
                    "text": " ".join(tks),
                    "spans": [tuple([" ".join(tks[span[0] : span[1]]) for span in spans]) for spans in gt_rl_spans],
                }

                continue

            if not gt_rl_spans:
                fp[relation_type] += len(pred_rl_spans)

                fp_report[relation_type][idx] = {
                    "text": " ".join(tks),
                    "spans": [
                        tuple([" ".join(tks[span[0] : span[1]]) for span in [sp[0] for sp in spans]])
                        for spans in pred_rl_spans
                    ],
                }

                continue

            # Calculate true positives, false positives, and false negatives for each entity type
            for pred_spans in pred_rl_spans:
                pred_cn_spans = pred_spans[0]
                pred_pn_spans = pred_spans[1]
                pred_pv_spans = pred_spans[2]

                cn_matched = False
                pn_matched = False
                pv_matched = False

                for pred_cn_span in pred_cn_spans:
                    if pred_cn_span in gt_cn_spans or (
                        allow_partial_match and any(overlaps(pred_cn_span, gt_cn_span) for gt_cn_span in gt_cn_spans)
                    ):
                        cn_matched = True
                        break
                if not cn_matched:
                    fp[relation_type] += 1
                    write_report(fp_report, relation_type, idx, (pred_cn_spans[0], pred_pn_spans[0], pred_pv_spans[0]))
                    continue

                for pred_pn_span in pred_pn_spans:
                    if pred_pn_span in gt_pn_spans or (
                        allow_partial_match and any(overlaps(pred_pn_span, gt_pn_span) for gt_pn_span in gt_pn_spans)
                    ):
                        pn_matched = True
                        break
                if not pn_matched:
                    fp[relation_type] += 1
                    write_report(fp_report, relation_type, idx, (pred_cn_spans[0], pred_pn_spans[0], pred_pv_spans[0]))
                    continue

                for pred_pv_span in pred_pv_spans:
                    if pred_pv_span in gt_pv_spans or (
                        allow_partial_match and any(overlaps(pred_pv_span, gt_pv_span) for gt_pv_span in gt_pv_spans)
                    ):
                        pv_matched = True
                        break
                if not pv_matched:
                    fp[relation_type] += 1
                    write_report(fp_report, relation_type, idx, (pred_cn_spans[0], pred_pn_spans[0], pred_pv_spans[0]))
                    continue

                tp[relation_type] += 1
                write_report(tp_report, relation_type, idx, (pred_cn_spans[0], pred_pn_spans[0], pred_pv_spans[0]))

            for gt_cn_span, gt_pn_span, gt_pv_span in zip(gt_cn_spans, gt_pn_spans, gt_pv_spans):
                if not (
                    (
                        gt_cn_span in pred_cn_spans
                        or (
                            allow_partial_match
                            and any(overlaps(gt_cn_span, pred_cn_span) for pred_cn_span in pred_cn_spans)
                        )
                    )
                    and (
                        gt_pn_span in pred_pn_spans
                        or (
                            allow_partial_match
                            and any(overlaps(gt_pn_span, pred_pn_span) for pred_pn_span in pred_pn_spans)
                        )
                    )
                    and (
                        gt_pv_span in pred_pv_spans
                        or (
                            allow_partial_match
                            and any(overlaps(gt_pv_span, pred_pv_span) for pred_pv_span in pred_pv_spans)
                        )
                    )
                ):
                    fn[relation_type] += 1
                    write_report(tp_report, relation_type, idx, (gt_cn_span, gt_pn_span, gt_pv_span))

    # Calculate precision, recall, and F1 score for each entity type
    metrics = {}
    for rl_type in relation_types:
        precision = tp[rl_type] / (tp[rl_type] + fp[rl_type] + 1e-9)
        recall = tp[rl_type] / (tp[rl_type] + fn[rl_type] + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        metrics[rl_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[rl_type],
            "fp": fp[rl_type],
            "fn": fn[rl_type],
        }

    # Calculate micro-averaged metrics
    tp = sum(tp.values())
    fp = sum(fp.values())
    fn = sum(fn.values())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    metrics["micro-average"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    reports = {"tp": tp_report, "fp": fp_report, "fn": fn_report}

    return metrics, reports
