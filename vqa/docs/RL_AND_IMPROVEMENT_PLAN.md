# RL And Improvement Plan

Last updated: 2026-04-29

## Requirement From The Course Spec

The optional advanced part in the project spec requires:

- Use reinforcement learning: PPO, DPO, or RLHF.
- Preference data: at least 100 pairs.
- Compare RL vs SFT using automatic metrics and human evaluation.

## Practical Direction For This Repo

Use B2 as the baseline SFT model, then add a preference-optimization stage.

Reason:

- A1/A2 are custom encoder-decoder models and are useful for the required architecture comparison.
- B2 is already a pretrained multimodal model with LoRA, so it is the most practical target for preference optimization.
- DPO is simpler and more stable than PPO for a student project because it trains from chosen/rejected pairs directly instead of requiring online rollouts and a value model.

Recommended advanced track:

1. SFT baseline: train B2 LoRA on `data/processed/annotations`.
2. Generate predictions from B2 and at least one other model.
3. Build at least 100 preference pairs.
4. Run DPO-style preference tuning or a lightweight supervised preference approximation.
5. Compare `B2-SFT` vs `B2-DPO/RL` on the same test split.
6. Add a small human evaluation table on 50-100 samples.

## Why DPO First

DPO is the lowest-risk option for this repo:

- It needs chosen/rejected answer pairs, which can be built from existing predictions.
- It avoids PPO instability and reward-model training overhead.
- It is aligned with the requirement because DPO is explicitly listed in the project spec.

PPO can be described as a stronger but more expensive alternative:

- Reward = weighted exact match / normalized text similarity / BERTScore / LLM judge.
- Need a reference policy and KL control.
- Needs more engineering with multimodal generation.

RLHF can be framed as the human-feedback variant:

- Collect human preferences on model answers.
- Train a reward model or use DPO directly from preferences.

## Preference Data Design

Minimum schema:

```json
{
  "preference_id": "pref_000001",
  "image": "data/processed/images/test/vts_000029.jpg",
  "image_id": "vts_000029",
  "question_id": "vts_000029_q01",
  "question": "Trong ảnh có biển cấm đi thẳng và rẽ phải không?",
  "chosen": "Có",
  "rejected": "don ' t know",
  "source": "gold_vs_model_prediction",
  "source_model": "B1_ZeroShot",
  "question_type": "yes_no"
}
```

Acceptable ways to create 100+ pairs:

- Gold-vs-wrong-model: `chosen = reference`, `rejected = incorrect prediction`.
- Better-model-vs-worse-model: chosen is a correct or more specific answer, rejected is wrong or vague.
- Human reviewed: manually mark chosen/rejected for ambiguous cases.

For the report, keep a CSV or JSONL sample of at least 100 pairs and describe annotation rules.

## Reward Function For PPO Or Analysis

Use a bounded reward in `[0, 1]`:

```text
reward = 0.60 * exact_match
       + 0.20 * rouge_l
       + 0.20 * meteor
```

Optional:

```text
reward = 0.50 * exact_match
       + 0.20 * bertscore_f1
       + 0.20 * llm_judge_score
       + 0.10 * answer_format_score
```

Answer format score:

- `1.0` if answer has at most 10 words.
- `0.5` if answer has 11-15 words.
- `0.0` otherwise.

## Experiments To Report

Main table:

| Model | Training | VQA Acc | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore | Human pref |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B2-SFT | LoRA supervised fine-tuning | | | | | | | |
| B2-DPO/RL | LoRA + preference tuning | | | | | | | |

Human eval:

- Sample 50-100 test rows.
- Show image, question, reference, B2-SFT prediction, B2-DPO/RL prediction.
- Human chooses which answer is better or marks tie.

## Implementation Phases

Phase 1: prepare preference data.

- Run B1/B2/A1/A2 evaluation with `--predictions-output`.
- Build `data/processed/preferences/preferences_v1.jsonl`.
- Manually inspect at least 100 pairs.

Phase 2: implement DPO path.

- Start with B2 BLIP LoRA as policy.
- Keep `checkpoints/model_b2/best_lora` as the SFT baseline.
- Train a second adapter into `checkpoints/model_b2_dpo`.

Phase 3: evaluate.

- Evaluate SFT and DPO/RL on identical test subset first.
- Then run full test if metrics improve or stay stable.

Phase 4: report.

- Include source citations for PPO, DPO, and RLHF.
- Explain why DPO was selected as the practical method.
- Show automatic metrics and human preference table.

## Sources

- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347.
- DPO: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", arXiv:2305.18290.
- RLHF/preferences: Christiano et al., "Deep Reinforcement Learning from Human Preferences", NeurIPS 2017.
- Implementation reference: Hugging Face TRL PPO/DPO documentation.
