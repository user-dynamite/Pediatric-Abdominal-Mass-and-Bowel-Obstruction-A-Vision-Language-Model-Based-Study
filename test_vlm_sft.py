import os, json, re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import joblib
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from peft import PeftModel

from dataset_vlm import VlmDataConfig, MultiModalReportDataset


@dataclass
class InferEvalCfg:
    ckpt_dir: str = "./outputs/Qwen_Qwen2.5-VL-7B-Instruct_fold4_best_f"
    base_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    root: str = "./data/NIA7/"
    test_csv: str = "./data/NIA7/All_data_fin_external.csv"
    text_col: str = "text"

    ct_k: int = 50
    ct_sampling: str = "uniform"
    use_ct_mosaic: bool = True
    mosaic_grid: tuple = (5, 10)

    use_tabular: bool = True

    max_length: int = 8192
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    batch_size: int = 1
    num_workers: int = 2
    use_4bit: bool = True
    bf16: bool = True

    # BERTScore 설정
    bertscore_lang: str = "en"   # 한국어면 "ko" 시도 가능(모델/환경에 따라 다름)

    out_jsonl: str = "external_test_generations.jsonl"
    out_metrics: str = "external_test_metrics.json"

    # 텍스트 정규화(공백/줄바꿈)
    normalize_whitespace: bool = True


cfg = InferEvalCfg()


def norm_text(s: str) -> str:
    s = s.strip()
    if cfg.normalize_whitespace:
        s = re.sub(r"\s+", " ", s)
    return s


def build_user_prompt(tab: Optional[Dict[str, float]]) -> str:
    base = (
        "You are a clinical assistant. Given the input medical images (X-ray and CT), "
        "generate the integrated diagnostic report in the required format.\n"
    )
    if tab is not None:
        base += (
            f"Patient info (normalized): age={tab['age']:.4f}, sex={tab['sex']:.4f}, "
            f"height={tab['height']:.4f}, weight={tab['weight']:.4f}\n"
        )
    base += "Return only the report."
    return base


def build_qwen_prompt_with_images(processor, user_text: str, n_images: int) -> str:
    user_content = [{"type": "image"} for _ in range(n_images)]
    user_content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": user_content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_from_ckpt_dir():
    processor = AutoProcessor.from_pretrained(cfg.ckpt_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt_dir, trust_remote_code=True)

    bnb = None
    if cfg.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        )

    base = AutoModelForVision2Seq.from_pretrained(
        cfg.base_model_name,
        trust_remote_code=True,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, cfg.ckpt_dir)
    model.eval()
    return model, processor, tokenizer


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    """
    원하는 지표:
    - ROUGE1/2/L
    - BLEU1/2 (corpus-level)
    - BERTScore (P/R/F1 평균)
    - METEOR
    """
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")          # n-gram별 BLEU 계산 편함
    meteor = evaluate.load("meteor")
    bert = evaluate.load("bertscore")

    # ROUGE
    rouge_res = rouge.compute(predictions=preds, references=refs)

    # BLEU-1/2
    # evaluate BLEU는 기본적으로 BLEU-4까지 계산하지만,
    # max_order를 바꿔서 BLEU-1, BLEU-2를 각각 계산할 수 있음
    bleu1 = bleu.compute(predictions=preds, references=refs, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=refs, max_order=2)["bleu"]

    # METEOR
    meteor_res = meteor.compute(predictions=preds, references=refs)

    # BERTScore (리스트로 나오므로 평균)
    bert_res = bert.compute(predictions=preds, references=refs, lang=cfg.bertscore_lang)
    bert_p = float(sum(bert_res["precision"]) / len(bert_res["precision"]))
    bert_r = float(sum(bert_res["recall"]) / len(bert_res["recall"]))
    bert_f1 = float(sum(bert_res["f1"]) / len(bert_res["f1"]))

    return {
        "rouge1": rouge_res.get("rouge1"),
        "rouge2": rouge_res.get("rouge2"),
        "rougeL": rouge_res.get("rougeL"),
        "rougeLsum": rouge_res.get("rougeLsum"),
        "bleu1": bleu1 * 100.0,  # bleu metric은 0~1이므로 보기 좋게 0~100 스케일로 변환
        "bleu2": bleu2 * 100.0,
        "meteor": meteor_res.get("meteor"),
        "bertscore_precision": bert_p,
        "bertscore_recall": bert_r,
        "bertscore_f1": bert_f1,
        "n": len(preds),
        "notes": {
            "bleu1_bleu2_scaled_to_0_100": True,
            "bertscore_lang": cfg.bertscore_lang,
            "normalize_whitespace": cfg.normalize_whitespace,
        }
    }


@torch.no_grad()
def main():
    scaler_path = os.path.join(cfg.ckpt_dir, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    model, processor, tokenizer = load_model_from_ckpt_dir()

    test_dc = VlmDataConfig(
        root=cfg.root,
        csv_path=cfg.test_csv,
        text_col=cfg.text_col,
        ct_k=cfg.ct_k,
        ct_sampling=cfg.ct_sampling,
        use_ct_mosaic=cfg.use_ct_mosaic,
        mosaic_grid=cfg.mosaic_grid,
        use_tabular=cfg.use_tabular,
        cache_images=False,
    )
    test_ds = MultiModalReportDataset(test_dc, split="test", scaler=scaler, fit_scaler=False)

    loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: b,
    )

    out_jsonl_path = os.path.join(cfg.ckpt_dir, cfg.out_jsonl)
    out_metrics_path = os.path.join(cfg.ckpt_dir, cfg.out_metrics)

    preds, refs = [], []

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="infer"):
            ex = batch[0]
            images = ex["images"]
            tab = ex.get("tabular", None)
            gt = ex["target"]
            pid = ex.get("id", None)

            if len(images) != 2:
                raise ValueError(f"Expected 2 images (X-ray + CT_mosaic), got {len(images)}")

            user_text = build_user_prompt(tab)
            prompt = build_qwen_prompt_with_images(processor, user_text, n_images=len(images))

            inputs = processor(
                text=[prompt],
                images=[images],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
            )

            dev = next(model.parameters()).device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(dev)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=(cfg.temperature if cfg.do_sample else None),
            )

            prompt_len = inputs["input_ids"].shape[1]
            out_ids = gen_ids[:, prompt_len:]
            pred = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

            pred_n = norm_text(pred)
            gt_n = norm_text(gt)

            preds.append(pred_n)
            refs.append(gt_n)

            f.write(json.dumps({"id": pid, "pred": pred_n, "gt": gt_n}, ensure_ascii=False) + "\n")

    metrics = compute_metrics(preds, refs)
    print("Metrics:\n", json.dumps(metrics, ensure_ascii=False, indent=2))

    with open(out_metrics_path, "w", encoding="utf-8") as wf:
        json.dump(metrics, wf, ensure_ascii=False, indent=2)

    print("Saved generations:", out_jsonl_path)
    print("Saved metrics:", out_metrics_path)


if __name__ == "__main__":
    main()
