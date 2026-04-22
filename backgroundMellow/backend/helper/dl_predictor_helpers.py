import os
import random
import sys
# Add project root to path
# import sys
# import os

# # Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# # Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)
print("Project root added to sys.path:", project_root)
    
# Cinemaudio-studio root (for tango_new when using Tango2)
cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
if cinema_studio_root not in sys.path:
    sys.path.append(cinema_studio_root)   
import json
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from model.dl_based_alignment_predictor import CinematicMixPredictor
from helper.audio_conversions import audio_to_base64
from helper.lib import ParlerTTSModel
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from helper.dl_conversions import normalize_targets
from typing import List

DATASET_PATH = "../data/yt_videos/generated_train_dataset.pkl"
logger = logging.getLogger(__name__)
YOUTUBE_DATASET_PATH = "./yt_dataset.jsonl"
MODEL_PATH = "model/dl_based_alignment_predictor.pth"
EMBEDDER_PATH = "model/embedder.pth"

# Save the dataset to a file for reuse to avoid recalculation each time
import pickle

def create_dataset():

    descriptions = [
        "A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting.",
        "A female speaker with a high-pitched voice is delivering her speech at a really fast speed in a noisy environment.",
        "A male speaker with a low-pitched voice is delivering his speech at a really slow speed in a quiet environment.",
        "A female speaker with a low-pitched voice is delivering her speech at a really fast speed in a noisy environment."
    ]
    datapath = YOUTUBE_DATASET_PATH
    dataset = []
    idx = 0
    with open(datapath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            story_prompt = data["story_prompt"]

            random_description = random.choice(descriptions)

            narrator_audio_segment = ParlerTTSModel.generate(prompt=story_prompt, description=random_description)
            narrator_audio_base64 = audio_to_base64(narrator_audio_segment)
            
            narrator_end_time = narrator_audio_segment.duration_seconds

            audio_classes = []
            clip_duration_sec = float(data.get("clip_duration_sec", 10.0))
            for audio_cues in data["cues"]:
                
                
                
                audio_classes.append({
                    "audio_class": audio_cues["audio_class"],
                    "weight_db": audio_cues["weight_db"],
                    "start_time_ms": audio_cues["starting_time"]/clip_duration_sec * narrator_end_time,
                    "duration_ms": audio_cues["duration"]/clip_duration_sec * narrator_end_time,
                })
                logger.info(f"entry {idx+1} audio_class: {audio_cues['audio_class']} start_time_ms: {audio_cues['starting_time']/clip_duration_sec * narrator_end_time} duration_ms: {audio_cues['duration']/clip_duration_sec * narrator_end_time}")
            
            
            dataset.append((story_prompt, audio_classes, narrator_audio_base64, clip_duration_sec))

            # Save dataset after each entry processed
            with open(DATASET_PATH, "wb") as dataset_file:
                pickle.dump(dataset, dataset_file)

            # Log the progress to indicate which entry has been processed
            logger.info(f"Processed and saved entry {idx+1}")
            idx += 1

    return dataset


def load_dataset():
    with open(DATASET_PATH, "rb") as dataset_file:
        return pickle.load(dataset_file)


def _collate_timeline_batch(batch):
    """Collate (story_emb, class_emb, timeline_seq, target) with padding for variable-length timeline."""
    story_embs = torch.stack([b[0] for b in batch])
    class_embs = torch.stack([b[1] for b in batch])
    timeline_seqs = [b[2] for b in batch]
    targets = torch.stack([b[3] for b in batch])

    max_T = max(seq.shape[0] for seq in timeline_seqs)
    embed_plus_3 = timeline_seqs[0].shape[1]
    device = timeline_seqs[0].device
    dtype = timeline_seqs[0].dtype

    padded = torch.zeros(len(batch), max_T, embed_plus_3, device=device, dtype=dtype)
    key_padding_mask = torch.ones(len(batch), max_T, dtype=torch.bool, device=device)
    for i, seq in enumerate(timeline_seqs):
        T_i = seq.shape[0]
        padded[i, :T_i] = seq
        key_padding_mask[i, :T_i] = False

    return story_embs, class_embs, padded, key_padding_mask, targets


class TimelineDataset(Dataset):
    """Dataset of (story_emb, class_emb, timeline_seq, target) for full-timeline alignment."""

    def __init__(self, samples: List[tuple]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_model(epochs=50, learning_rate=5e-4):
    dataset = load_dataset()
    logger.info("[train_model] Loaded dataset with %d entries", len(dataset))
    model = CinematicMixPredictor()

    samples = []
    num_stories = len(dataset)

    for story_idx, item in enumerate(dataset):
        if len(item) == 4:
            story_prompt, audio_classes, narrator_audio_base64, scene_duration = item
        else:
            story_prompt, audio_classes, narrator_audio_base64 = item
            scene_duration = 10.0
            logger.debug("[train_model] Entry has no clip_duration_sec; using default %.1fs", scene_duration)

        logger.info(
            "[train_model] Building examples for story %d/%d (%d cues), scene_duration=%.1fs",
            story_idx + 1, num_stories, len(audio_classes), scene_duration,
        )
        whisper_json = model.make_whisper_embedding(story_prompt, narrator_audio_base64)
        timeline_seq = model.get_whisper_sequence_tensor(whisper_json or [], scene_duration)

        story_emb = model.embedder.encode(
            story_prompt, convert_to_tensor=True, device=str(model.device)
        )

        for cue in audio_classes:
            audio_class = cue["audio_class"]
            class_emb = model.embedder.encode(
                audio_class, convert_to_tensor=True, device=str(model.device)
            )
            s_norm, w_norm, d_norm = normalize_targets(
                cue["start_time_ms"], cue["weight_db"], cue["duration_ms"],
            )
            y = torch.tensor([s_norm, w_norm, d_norm], dtype=torch.float32, device=model.device)
            samples.append((story_emb, class_emb, timeline_seq, y))

    train_dataset = TimelineDataset(samples)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=_collate_timeline_batch,
    )

    logger.info("[train_model] Built dataset: N=%d samples (full timeline per sample)", len(samples))
    if samples:
        t_shape = samples[0][2].shape
        logger.info("[train_model] Timeline shape per sample: %s (seq_len, embed_dim+3)", tuple(t_shape))
        targets_stack = torch.stack([s[3] for s in samples])
        logger.info(
            "[train_model] Normalized target ranges (min/max): start=%s, weight_db=%s, duration=%s",
            (targets_stack[:, 0].min().item(), targets_stack[:, 0].max().item()),
            (targets_stack[:, 1].min().item(), targets_stack[:, 1].max().item()),
            (targets_stack[:, 2].min().item(), targets_stack[:, 2].max().item()),
        )

    logger.info("[train_model] Starting training for %d epochs, lr=%s", epochs, learning_rate)
    losses = model.train_model(train_loader, epochs, learning_rate)
    model.save_model(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
    plt.tight_layout()
    plt.savefig("loss_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved loss curve to loss_curve.png")

    print("Model saved to", MODEL_PATH)
    



# train model
create_dataset()
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_model(epochs=100, learning_rate=5e-4)
    