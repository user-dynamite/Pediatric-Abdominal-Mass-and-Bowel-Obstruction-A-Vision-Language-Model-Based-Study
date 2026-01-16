import os, math, random, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import get_peft_model_state_dict
from accelerate import Accelerator

import joblib
import evaluate

from dataset_vlm import VlmDataConfig, MultiModalReportDataset


# -------------------------
# Config
# -------------------------
@dataclass
class TrainCfg:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "./outputs"

    # data
    root: str = "./data/NIA7/"
    train_csv: str = "./data/NIA7/fold_5_train.csv"
    valid_csv: str = "./data/NIA7/fold_5_val.csv"
    text_col: str = "text"

    # CT
    ct_k: int = 50
    ct_sampling: str = "uniform"
    use_ct_mosaic: bool = True
    mosaic_grid: tuple = (5, 10)

    # tabular
    use_tabular: bool = True

    # train
    seed: int = 42
    max_length: int = 8192
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_accum: int = 4
    per_device_batch: int = 4
    num_workers: int = 4

    # precision/quant
    bf16: bool = True
    use_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- Warmup ----
    use_warmup: bool = True
    warmup_ratio: float = 0.05
    warmup_steps_min: int = 200

    # ---- ReduceLROnPlateau (생성 metric 기준) ----
    # ROUGE-L이 더 이상 좋아지지 않으면 lr 감소
    plateau_factor: float = 0.5
    plateau_patience: int = 8
    plateau_threshold: float = 1e-4
    plateau_min_lr: float = 1e-6

    # ---- Early Stopping (생성 metric 기준) ----
    early_stop_patience: int = 5
    early_stop_min_delta: float = 5e-5

    # ---- Generation metric eval ----
    gen_eval_every: int = 5          # 매 epoch마다
    gen_eval_samples: int = 20       # val에서 일부만 샘플링 (속도/비용 절충)
    rouge_key: str = "rougeL"        # best 기준

    # ---- Decoding (beam search) ----
    max_new_tokens: int = 512
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.1


cfg = TrainCfg()


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# -------------------------
# Collator
# -------------------------
class VlmCollator:
    """
    Qwen2.5-VL은 이미지 placeholder를 messages에 {"type":"image"}로 넣어야 함.
    """
    def __init__(self, processor, tokenizer, max_length: int, model_name: str, expect_n_images: int = 2):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name.lower()
        self.expect_n_images = expect_n_images

    def _qwen_chat_text(self, user_text: str, assistant_text: Optional[str], n_images: int) -> str:
        user_content = [{"type": "image"} for _ in range(n_images)]
        user_content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": user_content}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(assistant_text is None)
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts, full_texts, images_list = [], [], []
        for ex in batch:
            user_text = build_user_prompt(ex.get("tabular"))
            n_images = len(ex["images"])
            if n_images != self.expect_n_images:
                raise ValueError(f"Expected {self.expect_n_images} images, got {n_images}. (Need X-ray + CT_mosaic)")
            prompts.append(self._qwen_chat_text(user_text, None, n_images))
            full_texts.append(self._qwen_chat_text(user_text, ex["target"], n_images))
            images_list.append(ex["images"])

        enc_full = self.processor(
            text=full_texts,
            images=images_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_prompt = self.processor(
            text=prompts,
            images=images_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = enc_full["input_ids"].clone()
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)
        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100

        out = {
            "input_ids": enc_full["input_ids"],
            "attention_mask": enc_full["attention_mask"],
            "labels": labels,
        }
        for k, v in enc_full.items():
            if k not in out:
                out[k] = v
        return out


# -------------------------
# Model loader
# -------------------------
def load_model(model_name: str):
    trust_remote_code = True

    bnb = None
    if cfg.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map=None,
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 요청대로 LLM attention proj에만 LoRA 적용
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, peft_cfg)

    return model, processor, tokenizer


# -------------------------
# Generation eval (ROUGE-L)
# -------------------------
@torch.no_grad()
def eval_gen_rougeL(
    model,
    processor,
    tokenizer,
    accelerator: Accelerator,
    valid_ds: MultiModalReportDataset,
    sample_indices: List[int],
) -> float:
    """
    val 일부 샘플에서 beam search로 생성 -> ROUGE-L 계산
    """
    rouge = evaluate.load("rouge")

    preds, refs = [], []
    model.eval()

    # sample_indices 순회 (batch=1로 단순하게)
    for idx in tqdm(sample_indices, desc="gen-eval", disable=not accelerator.is_main_process):
        ex = valid_ds[idx]
        images = ex["images"]
        tab = ex.get("tabular", None)
        gt = ex["target"]

        if len(images) != 2:
            raise ValueError(f"Expected 2 images (X-ray + CT_mosaic), got {len(images)}")

        user_text = build_user_prompt(tab)
        # Qwen: image placeholders
        user_content = [{"type": "image"} for _ in range(len(images))]
        user_content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": user_content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
            num_beams=cfg.num_beams,
            do_sample=False,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            length_penalty=cfg.length_penalty,
        )

        prompt_len = inputs["input_ids"].shape[1]
        out_ids = gen_ids[:, prompt_len:]
        pred = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        preds.append(pred)
        refs.append(gt)

    model.train()

    rouge_res = rouge.compute(predictions=preds, references=refs)
    rougeL = float(rouge_res["rougeL"])
    return rougeL


# -------------------------
# Train
# -------------------------
def main():
    set_seed(cfg.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision="bf16" if cfg.bf16 else "no",
    )

    model, processor, tokenizer = load_model(cfg.model_name)

    train_dc = VlmDataConfig(
        root=cfg.root,
        csv_path=cfg.train_csv,
        text_col=cfg.text_col,
        ct_k=cfg.ct_k,
        ct_sampling=cfg.ct_sampling,
        use_ct_mosaic=cfg.use_ct_mosaic,
        mosaic_grid=cfg.mosaic_grid,
        use_tabular=cfg.use_tabular,
        cache_images=False,
    )
    valid_dc = VlmDataConfig(
        root=cfg.root,
        csv_path=cfg.valid_csv,
        text_col=cfg.text_col,
        ct_k=cfg.ct_k,
        ct_sampling="uniform",
        use_ct_mosaic=cfg.use_ct_mosaic,
        mosaic_grid=cfg.mosaic_grid,
        use_tabular=cfg.use_tabular,
        cache_images=False,
    )

    train_ds = MultiModalReportDataset(train_dc, split="train", scaler=None, fit_scaler=True)
    valid_ds = MultiModalReportDataset(valid_dc, split="valid", scaler=train_ds.scaler, fit_scaler=False)

    if accelerator.is_main_process:
        ex0 = train_ds[0]
        print(f"[DEBUG] len(train_ds[0]['images']) = {len(ex0['images'])} (expected 2: X-ray + CT_mosaic)")

    collator = VlmCollator(processor, tokenizer, cfg.max_length, cfg.model_name, expect_n_images=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.per_device_batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    model, optim, train_loader, valid_loader = accelerator.prepare(
        model, optim, train_loader, valid_loader
    )

    # 스케줄러: ROUGE-L이 정체되면 lr 감소 (mode="max")
    plateau_sched = ReduceLROnPlateau(
        optim,
        mode="max",
        factor=cfg.plateau_factor,
        patience=cfg.plateau_patience,
        threshold=cfg.plateau_threshold,
        threshold_mode="rel",
        min_lr=cfg.plateau_min_lr,
    )

    # warmup 계산(optimizer update step 기준)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_updates = steps_per_epoch * cfg.epochs
    warmup_steps = max(int(total_updates * cfg.warmup_ratio), cfg.warmup_steps_min) if cfg.use_warmup else 0
    base_lr = cfg.lr
    if cfg.use_warmup:
        for pg in optim.param_groups:
            pg["lr"] = 0.0

    os.makedirs(cfg.output_dir, exist_ok=True)

    # val loss는 모니터링만(선택)
    def eval_val_loss() -> float:
        model.eval()
        losses = []
        vbar = tqdm(valid_loader, desc="valid-loss", disable=not accelerator.is_main_process)
        for batch in vbar:
            with torch.no_grad():
                out = model(**batch)
            loss_val = accelerator.gather(out.loss.detach()).float().mean().item()
            losses.append(loss_val)
            if accelerator.is_main_process:
                vbar.set_postfix(val_loss=f"{loss_val:.4f}")
        model.train()
        return sum(losses) / max(1, len(losses))

    # gen-eval 샘플 인덱스 고정(매 epoch 동일 subset으로 비교)
    rng = random.Random(cfg.seed)
    all_idx = list(range(len(valid_ds)))
    rng.shuffle(all_idx)
    gen_eval_n = min(cfg.gen_eval_samples, len(valid_ds))
    gen_eval_indices = all_idx[:gen_eval_n]

    best_metric = -1e9  # ROUGE-L maximize
    best_epoch = -1
    no_improve_epochs = 0
    global_update_step = 0

    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch+1}/{cfg.epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)

                # (1) 누적 끝난 시점에만 업데이트
                if accelerator.sync_gradients:
                    # warmup은 update step 기준으로
                    if cfg.use_warmup and global_update_step < warmup_steps:
                        warmup_lr = base_lr * float(global_update_step + 1) / float(warmup_steps)
                        for pg in optim.param_groups:
                            pg["lr"] = warmup_lr

                    optim.step()
                    optim.zero_grad()
                    global_update_step += 1

            if accelerator.is_main_process:
                cur_lr = optim.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{cur_lr:.2e}",
                    upd=str(global_update_step),
                )

        # ---- epoch end: (선택) val loss 로깅 ----
        val_loss = eval_val_loss()
        if accelerator.is_main_process:
            print(f"[epoch {epoch}] val_loss={val_loss:.4f} | lr={optim.param_groups[0]['lr']:.2e}")

        # ---- (2) 생성 성능 평가(ROUGE-L) ----
        metric = None
        if (epoch + 1) % cfg.gen_eval_every == 0:
            metric = eval_gen_rougeL(model, processor, tokenizer, accelerator, valid_ds, gen_eval_indices)
            if accelerator.is_main_process:
                print(f"[epoch {epoch}] gen {cfg.rouge_key}={metric:.4f} (n={gen_eval_n})")

            # warmup 끝난 뒤에만 plateau 적용 권장
            if (not cfg.use_warmup) or (global_update_step >= warmup_steps):
                plateau_sched.step(metric)
                if accelerator.is_main_process:
                    print(f"  -> lr after plateau(metric): {optim.param_groups[0]['lr']:.2e}")

            # ---- (2) best 저장/early stop 기준을 생성 metric으로 ----
            improved = (metric - best_metric) > cfg.early_stop_min_delta
            if improved:
                best_metric = metric
                best_epoch = epoch
                no_improve_epochs = 0

                if accelerator.is_main_process:
                    print(f"  -> New best! {cfg.rouge_key}={best_metric:.4f} at epoch={best_epoch}. Saving...")

                    unwrapped = accelerator.unwrap_model(model)
                    save_dir = os.path.join(cfg.output_dir, f"{cfg.model_name.replace('/', '_')}_fold5_best")
                    os.makedirs(save_dir, exist_ok=True)

                    # HF adapter 저장(PT 직접 로드보다 안정적)
                    unwrapped.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)

                    # adapter.pt도 같이 저장(참고용)
                    adapter_sd = get_peft_model_state_dict(unwrapped)
                    torch.save(adapter_sd, os.path.join(save_dir, "adapter.pt"))

                    joblib.dump(train_ds.scaler, os.path.join(save_dir, "scaler.joblib"))

                    meta = {
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "metric_name": cfg.rouge_key,
                        "gen_eval_n": gen_eval_n,
                        "train_cfg": cfg.__dict__,
                    }
                    with open(os.path.join(save_dir, "train_meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

            else:
                no_improve_epochs += 1
                if accelerator.is_main_process:
                    print(f"  -> No improvement (metric): {no_improve_epochs}/{cfg.early_stop_patience}")

                if no_improve_epochs >= cfg.early_stop_patience:
                    if accelerator.is_main_process:
                        print(f"[EarlyStopping(metric)] Stop at epoch={epoch}. Best {cfg.rouge_key}={best_metric:.4f} at epoch={best_epoch}")
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[Done] Best {cfg.rouge_key}={best_metric:.4f} at epoch={best_epoch}")


if __name__ == "__main__":
    main()
