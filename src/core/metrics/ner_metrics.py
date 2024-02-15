import logging
from ..utils import overlaps

logger = logging.getLogger(__name__)


def get_ner_metrics(
    pred_list: list[dict[tuple[int, int], str]],
    gt_list: list[dict[tuple[int, int], str]],
    entity_types: list[str],
    tks_list: list[list[str]],
    ids_list: list[int] = None,
    allow_partial_match: bool = False,
):
    # Initialize dictionaries to store counts for each entity type
    tp = {ent_type: 0 for ent_type in entity_types}
    fp = {ent_type: 0 for ent_type in entity_types}
    fn = {ent_type: 0 for ent_type in entity_types}

    tp_report = {ent_type: dict() for ent_type in entity_types}
    fp_report = {ent_type: dict() for ent_type in entity_types}
    fn_report = {ent_type: dict() for ent_type in entity_types}

    for idx, pred, gt, tks in zip(ids_list, pred_list, gt_list, tks_list):
        # Convert predicted and ground_truth to sets for easier comparison
        pred_set = set(pred.items())
        gt_set = set(gt.items())

        # Convert ground_truth to a dictionary of spans to entity types for partial match checking
        gt_spans = {span: et for span, et in gt.items()}

        # Calculate true positives, false positives, and false negatives for each entity type
        for pred_span, pred_type in pred_set:
            if pred_type not in entity_types:
                continue

            if (pred_span, pred_type) in gt_set:
                tp[pred_type] += 1

                if tp_report[pred_type].get(idx, None) is None:
                    tp_report[pred_type][idx] = {
                        "text": " ".join(tks),
                        "pred": [" ".join(tks[pred_span[0] : pred_span[1]])],
                    }
                else:
                    tp_report[pred_type][idx]["pred"].append(" ".join(tks[pred_span[0] : pred_span[1]]))

            elif allow_partial_match and any(
                overlaps(pred_span, gt_span) and pred_type == gt_spans[gt_span] for gt_span in gt_spans
            ):
                tp[pred_type] += 1  # Counting partial matches as true positives

                if tp_report[pred_type].get(idx, None) is None:
                    tp_report[pred_type][idx] = {
                        "text": " ".join(tks),
                        "pred": [" ".join(tks[pred_span[0] : pred_span[1]])],
                    }
                else:
                    tp_report[pred_type][idx]["pred"].append(" ".join(tks[pred_span[0] : pred_span[1]]))

            else:
                fp[pred_type] += 1

                if fp_report[pred_type].get(idx, None) is None:
                    fp_report[pred_type][idx] = {
                        "text": " ".join(tks),
                        "pred": [" ".join(tks[pred_span[0] : pred_span[1]])],
                    }
                else:
                    fp_report[pred_type][idx]["pred"].append(" ".join(tks[pred_span[0] : pred_span[1]]))

        for gt_span, gt_type in gt_set:
            if gt_type not in entity_types:
                continue

            if not any(
                (gt_span, gt_type) == (pred_span, pred_type)
                or (allow_partial_match and overlaps(gt_span, pred_span) and gt_type == pred_type)
                for pred_span, pred_type in pred_set
            ):
                fn[gt_type] += 1

                if fn_report[gt_type].get(idx, None) is None:
                    fn_report[gt_type][idx] = {
                        "text": " ".join(tks),
                        "gt": [" ".join(tks[gt_span[0] : gt_span[1]])],
                    }
                else:
                    fn_report[gt_type][idx]["gt"].append(" ".join(tks[gt_span[0] : gt_span[1]]))

    # Calculate precision, recall, and F1 score for each entity type
    metrics = {}
    for pred_type in entity_types:
        precision = tp[pred_type] / (tp[pred_type] + fp[pred_type] + 1e-9)
        recall = tp[pred_type] / (tp[pred_type] + fn[pred_type] + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        metrics[pred_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[pred_type],
            "fp": fp[pred_type],
            "fn": fn[pred_type],
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
