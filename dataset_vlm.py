import os
import json
import glob
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def load_gray_as_rgb(path: str) -> Image.Image:
    # 너 코드는 'L'(grayscale)였지만, VLM processor는 보통 RGB 기대
    return Image.open(path).convert("RGB")


def uniform_sample_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    # 균일 샘플링
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def random_sample_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return random.sample(range(n), k)


def make_mosaic(pil_images: List[Image.Image], grid: Tuple[int, int]) -> Image.Image:
    """
    pil_images: list of RGB PIL images (all same size 권장)
    grid: (rows, cols)
    """
    if len(pil_images) == 0:
        raise ValueError("No images for mosaic")

    rows, cols = grid
    w, h = pil_images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))

    # 부족하면 마지막 이미지 반복
    imgs = pil_images[:]
    while len(imgs) < rows * cols:
        imgs.append(imgs[-1])

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        canvas.paste(imgs[idx], (c * w, r * h))
    return canvas


@dataclass
class VlmDataConfig:
    root: str                    # data_path
    csv_path: str                # root + csv (또는 절대경로)
    ct_glob: str = "02. CT/*.png"
    cr_glob: str = "01. CR/*.png"

    # CT 샘플링
    ct_k: int = 50
    ct_sampling: str = "uniform"   # "uniform" or "random"

    # CT mosaic
    use_ct_mosaic: bool = True
    mosaic_grid: Tuple[int, int] = (5, 10)  # ct_k=8일 때 2x4

    # tabular
    use_tabular: bool = True
    tabular_cols: Tuple[str, ...] = ("age", "sex", "height", "weight")

    # 텍스트 컬럼
    text_col: str = "text"
    id_col: str = "ID"

    # 전처리/캐시
    cache_images: bool = False  # VLM 학습은 메모리 부담 커서 기본 False 권장


class MultiModalReportDataset(Dataset):
    """
    너가 준 CustomDataset 흐름:
    - CSV 읽고
    - sex 매핑
    - (선택) MinMaxScaler
    - patient_id로 CR/CT png 로드
    - CT는 K장 샘플링
    - 반환: images(list[PIL]), target(str), tabular(dict/tensor)
    """
    def __init__(
        self,
        cfg: VlmDataConfig,
        split: str,
        scaler: Optional[MinMaxScaler] = None,
        fit_scaler: bool = False,
    ):
        self.cfg = cfg
        self.split = split

        self.df = pd.read_csv(cfg.csv_path)
        # sex 매핑
        if "sex" in self.df.columns:
            self.df["sex"] = self.df["sex"].map({"M": 0, "F": 1}).astype(float)

        # scaler 처리 (중요: train에서만 fit)
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        if cfg.use_tabular:
            X = self.df[list(cfg.tabular_cols)].copy()
            if fit_scaler:
                self.scaler.fit(X)
            self.df[list(cfg.tabular_cols)] = self.scaler.transform(X)

        self._cache = []
        if cfg.cache_images:
            for i in range(len(self.df)):
                self._cache.append(self._load_item(i))

    def __len__(self):
        return len(self.df)

    def _get_ct_paths(self, patient_id: str) -> List[str]:
        paths = glob.glob(os.path.join(self.cfg.root, str(patient_id), self.cfg.ct_glob))
        paths = sorted(paths)
        return paths

    def _get_cr_path(self, patient_id: str) -> str:
        paths = glob.glob(os.path.join(self.cfg.root, str(patient_id), self.cfg.cr_glob))
        paths = sorted(paths)
        if len(paths) == 0:
            raise FileNotFoundError(f"No CR found for patient {patient_id}")
        return paths[0]

    def _load_item(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        patient_id = row[self.cfg.id_col]

        # CR
        cr_path = self._get_cr_path(patient_id)
        cr_img = load_gray_as_rgb(cr_path)

        # CT (K slices)
        ct_paths = self._get_ct_paths(patient_id)
        if len(ct_paths) == 0:
            # CT가 없는 케이스도 있다면 정책 필요 (여기선 CR만)
            ct_imgs = []
        else:
            if self.cfg.ct_sampling == "random":
                idxs = random_sample_indices(len(ct_paths), self.cfg.ct_k)
            else:
                idxs = uniform_sample_indices(len(ct_paths), self.cfg.ct_k)

            ct_imgs = [load_gray_as_rgb(ct_paths[i]) for i in idxs]

        # CT를 mosaic 1장으로 합치기(권장: 모델간 공정 비교 쉬움)
        images: List[Image.Image] = [cr_img]
        if len(ct_imgs) > 0:
            if self.cfg.use_ct_mosaic:
                # ct_mosaic 만든 직후
                ct_mosaic = make_mosaic(ct_imgs, self.cfg.mosaic_grid)

                # 매우 중요: 모자이크가 너무 크면 vision token 폭증 -> 길이/메모리 터짐
                ct_mosaic = ct_mosaic.resize((512, 512))   # 448 또는 384 추천
                cr_img = cr_img.resize((512, 512))

                images.append(ct_mosaic)
            else:
                # multi-image로 그대로 넣고 싶을 때
                images.extend(ct_imgs)

        target = str(row[self.cfg.text_col])

        tab = None
        if self.cfg.use_tabular:
            tab = {c: float(row[c]) for c in self.cfg.tabular_cols}

        return {
            "id": str(patient_id),
            "images": images,   # list[PIL]
            "target": target,   # str
            "tabular": tab,     # dict or None
        }

    def __getitem__(self, index):
        if self.cfg.cache_images:
            return self._cache[index]
        return self._load_item(index)
