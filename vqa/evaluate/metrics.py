import math
import re
from collections import Counter


def normalize_answer(ans):
    ans = str(ans or "").lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)
    return " ".join(ans.split())


def _ngrams(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(predictions, references, max_order):
    matches = [0] * max_order
    possible = [0] * max_order
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_ngrams = _ngrams(pred_tokens, order)
            ref_ngrams = _ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches[order - 1] += sum(overlap.values())
            possible[order - 1] += max(len(pred_tokens) - order + 1, 0)

    if pred_len == 0:
        return 0.0

    precisions = []
    for i in range(max_order):
        if possible[i] == 0 or matches[i] == 0:
            return 0.0
        precisions.append(matches[i] / possible[i])
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return brevity_penalty * geo_mean


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        if pred == ref:
            scores.append(1.0)
            continue
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs = _lcs_len(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def _meteor(predictions, references):
    """Lightweight METEOR-style score without external corpora downloads.

    This uses exact unigram matches, the standard METEOR precision/recall
    weighting, and a simple fragmentation penalty. It is deterministic and works
    for Vietnamese tokenized by whitespace.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        if pred == ref:
            scores.append(1.0)
            continue
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        ref_used = [False] * len(ref_tokens)
        matched_ref_positions = []
        matches = 0
        for token in pred_tokens:
            for idx, ref_token in enumerate(ref_tokens):
                if not ref_used[idx] and token == ref_token:
                    ref_used[idx] = True
                    matched_ref_positions.append(idx)
                    matches += 1
                    break

        if matches == 0:
            scores.append(0.0)
            continue

        precision = matches / len(pred_tokens)
        recall = matches / len(ref_tokens)
        f_mean = (10 * precision * recall) / (recall + 9 * precision)

        chunks = 1
        for prev, curr in zip(matched_ref_positions, matched_ref_positions[1:]):
            if curr != prev + 1:
                chunks += 1
        penalty = 0.5 * (chunks / matches) ** 3
        scores.append(f_mean * (1 - penalty))

    return sum(scores) / len(scores) if scores else 0.0


def _bert_score(predictions, references, model_type="xlm-roberta-base",
                device=None):
    try:
        from bert_score import score

        _, _, f1 = score(
            predictions,
            references,
            model_type=model_type,
            lang="vi",
            verbose=False,
            device=device,
        )
        return float(f1.mean().item())
    except Exception as exc:
        return {"error": str(exc)}


def compute_metrics(predictions, references, compute_bertscore=False,
                    bertscore_model="xlm-roberta-base", device=None):
    norm_preds = [normalize_answer(p) for p in predictions]
    norm_refs = [normalize_answer(r) for r in references]

    if not norm_refs:
        return {
            "bleu1": 0.0,
            "bleu4": 0.0,
            "rouge_l": 0.0,
            "meteor": 0.0,
            "bertscore_f1": None,
            "vqa_accuracy": 0.0,
        }

    metrics = {
        "bleu1": _corpus_bleu(norm_preds, norm_refs, 1),
        "bleu4": _corpus_bleu(norm_preds, norm_refs, 4),
        "rouge_l": _rouge_l(norm_preds, norm_refs),
        "meteor": _meteor(norm_preds, norm_refs),
        "bertscore_f1": None,
        "vqa_accuracy": sum(p == r for p, r in zip(norm_preds, norm_refs)) / len(norm_refs),
    }

    if compute_bertscore:
        value = _bert_score(
            predictions,
            references,
            model_type=bertscore_model,
            device=device,
        )
        if isinstance(value, dict):
            metrics["bertscore_error"] = value["error"]
        else:
            metrics["bertscore_f1"] = value

    return metrics
