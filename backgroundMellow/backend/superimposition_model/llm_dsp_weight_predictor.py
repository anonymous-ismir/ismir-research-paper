"""
Tiny DL-based predictor for the LLM vs DSP blend factor (alpha).

Target rule (from project prompt/mix hierarchy):
- MUSIC / MOVIE_BGM: trust DSP more -> alpha close to 1.0 (e.g. 0.8)
- SFX / AMBIENCE / NARRATOR: trust LLM more -> alpha close to 0.0 (e.g. 0.2)

This module trains a very small MLP on a synthetic dataset encoding the above
rules and saves it to `backend/model/llm_dsp_weight_predictor.pth`.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.adam import Adam

# Local copy to avoid importing `Variable.configurations`, which pulls in
# optional audio deps (e.g. pydub) via `headers.imports`.
SOUND_TYPES = ["SFX", "AMBIENCE", "MUSIC", "NARRATOR", "MOVIE_BGM"]


MODEL_FILENAME = "llm_dsp_weight_predictor.pth"
MUSIC_AUDIO_TYPES = {"MUSIC", "MOVIE_BGM"}

# Kept intentionally small: model only needs to learn a single scalar.
_DEFAULT_HIDDEN_DIM = 8
_DEFAULT_LEARNING_RATE = 1e-2
_DEFAULT_EPOCHS = 200

_MUSIC_KEYWORDS = [
    "music",
    "orchestral",
    "melody",
    "theme",
    "score",
    "soundtrack",
    "emotional",
]

_NARRATION_KEYWORDS = [
    "narration",
    "narrator",
    "dialogue",
    "voice",
    "speaks",
]

_SFX_KEYWORDS = [
    "sfx",
    "impact",
    "footstep",
    "footsteps",
    "door",
    "whoosh",
    "blast",
    "gun",
    "reload",
    "click",
]

_AMBIENCE_KEYWORDS = [
    "rain",
    "wind",
    "ambience",
    "crowd",
    "forest",
    "room tone",
    "rumble",
]


def _get_model_path() -> str:
    # backend/superimposition_model/llm_dsp_weight_predictor.py -> backend/model/llm_dsp_weight_predictor.pth
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(backend_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, MODEL_FILENAME)


def _set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _audio_class_keyword_score(audio_class: str) -> float:
    """
    Simple numeric feature to mirror the prompt rule:
    - high if the cue text looks musical
    - low otherwise
    """
    s = (audio_class or "").lower()
    if any(k in s for k in _MUSIC_KEYWORDS):
        return 1.0
    # If it explicitly looks like narration/SFX/ambience, keep it low.
    if any(k in s for k in _NARRATION_KEYWORDS + _SFX_KEYWORDS + _AMBIENCE_KEYWORDS):
        return 0.0
    # Unknown cue text: keep neutral-ish.
    return 0.5


def _base_alpha_for_audio_type(audio_type: str) -> float:
    if (audio_type or "").upper() in MUSIC_AUDIO_TYPES:
        return 0.8
    return 0.2


class LlmDspAlphaMLP(nn.Module):
    """
    Very small MLP that outputs alpha in [0, 1] via sigmoid.

    Input features:
    - one-hot audio_type over SOUND_TYPES
    - 1 scalar keyword score derived from audio_class text
    """

    def __init__(self, audio_type_to_idx: Dict[str, int], hidden_dim: int = _DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.audio_type_to_idx = dict(audio_type_to_idx)
        num_types = len(self.audio_type_to_idx)
        input_dim = num_types + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim] -> alpha: [B, 1]
        return torch.sigmoid(self.net(x))


@dataclass
class BlendAlphaPredictor:
    model: LlmDspAlphaMLP
    audio_type_to_idx: Dict[str, int]
    device: torch.device

    @torch.no_grad()
    def predict_alpha(self, audio_type: str, audio_class: str = "") -> float:
        """
        Predict alpha where:
        blended = (1-alpha)*LLM + alpha*DSP
        """
        audio_type_key = (audio_type or "").upper().strip()
        onehot = torch.zeros(len(self.audio_type_to_idx), device=self.device, dtype=torch.float32)
        idx = self.audio_type_to_idx.get(audio_type_key)
        if idx is not None:
            onehot[idx] = 1.0

        keyword_score = torch.tensor(
            [_audio_class_keyword_score(audio_class)],
            device=self.device,
            dtype=torch.float32,
        )
        x = torch.cat([onehot, keyword_score], dim=0).unsqueeze(0)  # [1, D]
        alpha = float(self.model(x).squeeze(0).squeeze(-1).item())
        # Numerical safety
        return max(0.0, min(1.0, alpha))


_PREDICTOR: Optional[BlendAlphaPredictor] = None


def _build_synthetic_dataset(
    samples_per_type: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Build a fake dataset that encodes your rule:
    - MUSIC/MOVIE_BGM -> alpha ~ 0.8
    - else -> alpha ~ 0.2
    """
    audio_types = [t.upper() for t in SOUND_TYPES]
    audio_type_to_idx = {t: i for i, t in enumerate(audio_types)}

    # Make fake audio_class strings that align with the keyword feature.
    templates = {
        "MUSIC": "Orchestral emotional music theme with melodic score",
        "MOVIE_BGM": "Cinematic soundtrack theme music cue",
        "SFX": "Sharp impact SFX footsteps door whoosh blast reload click",
        "AMBIENCE": "Rain ambience wind ambience forest room tone rumble",
        "NARRATOR": "Narration dialogue voice speaks in scene",
    }

    X: List[torch.Tensor] = []
    y: List[torch.Tensor] = []

    for audio_type in audio_types:
        base_alpha = _base_alpha_for_audio_type(audio_type)
        for _ in range(samples_per_type):
            # Add small label noise so the model learns a smooth mapping.
            alpha = max(0.05, min(0.95, base_alpha + random.uniform(-0.05, 0.05)))

            audio_class = templates.get(audio_type, "Generic cue")
            keyword_score = _audio_class_keyword_score(audio_class)

            onehot = torch.zeros(len(audio_type_to_idx), dtype=torch.float32)
            onehot[audio_type_to_idx[audio_type]] = 1.0

            # Add tiny input noise (helps prevent overly brittle fits).
            # Keep one-hot near 0/1 by only adding noise then clamping.
            noisy_onehot = torch.clamp(onehot + 0.01 * torch.randn_like(onehot), 0.0, 1.0)
            noisy_keyword = float(max(0.0, min(1.0, keyword_score + random.uniform(-0.05, 0.05))))

            feats = torch.cat([noisy_onehot, torch.tensor([noisy_keyword], dtype=torch.float32)], dim=0)
            X.append(feats)
            y.append(torch.tensor([alpha], dtype=torch.float32))

    X_t = torch.stack(X, dim=0)
    y_t = torch.stack(y, dim=0)
    return X_t, y_t, audio_type_to_idx


def train_fake_and_save(
    path: Optional[str] = None,
    *,
    samples_per_type: int = 300,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    lr: float = _DEFAULT_LEARNING_RATE,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 1337,
) -> str:
    """
    Train on a synthetic dataset encoding the desired DSP/LLM weighting rules.
    Saves a checkpoint and returns its path.
    """
    _set_seed(seed)
    ckpt_path = path or _get_model_path()
    device = _device()

    X, y, audio_type_to_idx = _build_synthetic_dataset(samples_per_type=samples_per_type)
    X = X.to(device)
    y = y.to(device)

    model = LlmDspAlphaMLP(audio_type_to_idx=audio_type_to_idx, hidden_dim=hidden_dim).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean")

    batch_size = 64
    n = X.shape[0]

    # Shuffle indices once per epoch for simplicity.
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        X_shuf = X[perm]
        y_shuf = y[perm]

        total_loss = 0.0
        num_batches = 0
        for i in range(0, n, batch_size):
            xb = X_shuf[i : i + batch_size]
            yb = y_shuf[i : i + batch_size]

            pred = model(xb)
            loss = mse(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

    # Save a lightweight checkpoint.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "audio_type_to_idx": audio_type_to_idx,
        "hidden_dim": hidden_dim,
        "device": str(device),
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def load_predictor(
    path: Optional[str] = None,
    *,
    train_if_missing: bool = True,
) -> BlendAlphaPredictor:
    """
    Load the trained predictor, optionally training a new model if missing.
    """
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    ckpt_path = path or _get_model_path()
    device = _device()

    if not os.path.exists(ckpt_path):
        if not train_if_missing:
            raise FileNotFoundError(f"Missing predictor checkpoint: {ckpt_path}")
        train_fake_and_save(path=ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    audio_type_to_idx = checkpoint["audio_type_to_idx"]
    hidden_dim = int(checkpoint.get("hidden_dim", _DEFAULT_HIDDEN_DIM))

    model = LlmDspAlphaMLP(audio_type_to_idx=audio_type_to_idx, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    _PREDICTOR = BlendAlphaPredictor(
        model=model,
        audio_type_to_idx=audio_type_to_idx,
        device=device,
    )
    return _PREDICTOR


def predict_alpha_for_audio_cue(audio_type: str, audio_class: str) -> float:
    """
    Convenience function used by alignment code.
    """
    predictor = load_predictor(train_if_missing=True)
    return predictor.predict_alpha(audio_type=audio_type, audio_class=audio_class)


# if __name__ == "__main__":
#     ckpt = train_fake_and_save()
#     print(f"Saved llm_dsp_weight_predictor checkpoint to: {ckpt}")
    
#     # test the predictor
#     audio_type = "MUISC"
#     audio_class = "storytelling music"
#     alpha = predict_alpha_for_audio_cue(audio_type=audio_type, audio_class=audio_class)
#     print(f"Alpha for {audio_type} {audio_class}: {alpha}")
    
