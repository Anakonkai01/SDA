import re
from evaluate import load

bleu_metric = load("bleu")
rouge_metric = load("rouge")
bertscore_metric = load("bertscore")


def normalize_answer(ans):
    ans = ans.lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)
    return " ".join(ans.split())


def compute_metrics(predictions, references):
    norm_preds = [normalize_answer(p) for p in predictions]
    norm_refs = [normalize_answer(r) for r in references]

    results = {}

    results["bleu1"] = bleu_metric.compute(
        predictions=norm_preds,
        references=[[r] for r in norm_refs],
        max_order=1,
    )["bleu"]

    results["bleu4"] = bleu_metric.compute(
        predictions=norm_preds,
        references=[[r] for r in norm_refs],
        max_order=4,
    )["bleu"]

    results["rouge_l"] = rouge_metric.compute(
        predictions=norm_preds,
        references=norm_refs,
    )["rougeL"]

    bs = bertscore_metric.compute(
        predictions=norm_preds,
        references=norm_refs,
        lang="vi",
    )
    results["bertscore_f1"] = sum(bs["f1"]) / len(bs["f1"])

    results["vqa_accuracy"] = sum(
        p == r for p, r in zip(norm_preds, norm_refs)
    ) / len(norm_preds)

    return results
