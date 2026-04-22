# import os
# import sys
# # Add project root to path
# # import sys
# # import os

# # # Get absolute path of project root (one level up from current notebook)
# project_root = os.path.abspath("..")

# # # Add to sys.path if not already
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print("Project root added to sys.path:", project_root)
    
# # Cinemaudio-studio root (for tango_new when using Tango2)
# cinema_studio_root = os.path.abspath(os.path.join(project_root, ".."))
# if cinema_studio_root not in sys.path:
#     sys.path.append(cinema_studio_root)    


import base64
import io
import json
import logging
import os
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from sentence_transformers import SentenceTransformer
from model.word_aligner import WordAligner

logger = logging.getLogger(__name__)

MODEL_PATH = "model/dl_based_alignment_predictor.pth"

from helper.dl_conversions import denormalize_outputs
class CinematicMixPredictor(nn.Module):
    """
    Full-timeline cross-attention: tokens [Story, Class, Word_1, ..., Word_N].
    Class token attends over the whole narrative timeline; we take its output
    (+ residual class_emb) and predict [start_time, weight_db, duration].
    No rule-based BGM/SFX anchoring — the model learns when to place each sound.
    """

    def __init__(self, embed_dim=384, n_heads=4, attention_dropout=0.1):
        super(CinematicMixPredictor, self).__init__()

        self.embed_dim = embed_dim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.n_heads = n_heads

        # Project timeline tokens [word_embedding + start, end, scene_duration] -> embed_dim
        self.timeline_project = nn.Linear(embed_dim + 3, embed_dim)

        self.story_ln = nn.LayerNorm(embed_dim)
        self.class_ln = nn.LayerNorm(embed_dim)
        self.timeline_ln = nn.LayerNorm(embed_dim)

        # Self-attention: class embedding, story embedding, and timeline (word) tokens attend to each other
        self.self_attn_1 = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm_1 = nn.LayerNorm(embed_dim)
        # Second attention block for deeper interaction
        self.self_attn_2 = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm_2 = nn.LayerNorm(embed_dim)

        self.head_ln = nn.LayerNorm(embed_dim)
        # FFN on class token output -> [start_time, weight_db, duration]
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Load existing weights only if checkpoint exists. Filter by shape for compatibility
        # with older checkpoints (e.g. anchor_project vs timeline_project).
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            except Exception:
                state = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            if not isinstance(state, dict):
                logger.warning("Checkpoint is not a state_dict; skipping load.")
            else:
                model_state = self.state_dict()
                filtered = {
                    k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                skipped = [k for k in state if k not in filtered]
                if skipped:
                    logger.warning(
                        "Checkpoint: skipped keys (shape mismatch or missing): %s",
                        skipped,
                    )
                self.load_state_dict(filtered, strict=False)
                logger.info("Loaded alignment predictor weights from %s", MODEL_PATH)
        else:
            logger.info("No checkpoint at %s; using randomly initialized weights.", MODEL_PATH)

        self.to(self.device)

    def forward(
        self,
        story_emb: torch.Tensor,
        class_emb: torch.Tensor,
        timeline_seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict for a single audio cue (or batch of cues). Self-attention between
        class embedding, story embedding, and timeline word tokens.

        story_emb: [embed_dim] or [batch, embed_dim]
        class_emb: [embed_dim] or [batch, embed_dim]
        timeline_seq: [seq_len, embed_dim + 3] or [batch, seq_len, embed_dim + 3]
        key_padding_mask: [seq_len] or [batch, seq_len] True = ignore. Optional.
        Returns: [3] or [batch, 3] (start_time, weight_db, duration) normalized in [-1, 1].
        """
        
        single = story_emb.dim() == 1
        if single:
            story_emb = story_emb.unsqueeze(0)
            class_emb = class_emb.unsqueeze(0)
            timeline_seq = timeline_seq.unsqueeze(0)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # Project timeline: [B, T, embed_dim+3] -> [B, T, embed_dim]
        timeline_proj = self.timeline_project(timeline_seq)
        story_tok = self.story_ln(story_emb).unsqueeze(1)   # [B, 1, embed_dim]
        class_tok = self.class_ln(class_emb).unsqueeze(1)   # [B, 1, embed_dim]
        timeline_proj = self.timeline_ln(timeline_proj)

        # Sequence: [Story, Class, Word_1, ..., Word_N] — all attend to each other
        tokens = torch.cat([story_tok, class_tok, timeline_proj], dim=1)  # [B, 2+T, embed_dim]

        if key_padding_mask is not None:
            pad_prefix = torch.zeros(
                key_padding_mask.shape[0], 2, dtype=torch.bool, device=key_padding_mask.device
            )
            key_padding_mask = torch.cat([pad_prefix, key_padding_mask], dim=1)

        # Block 1: self-attention (class, story, timeline tokens)
        attn_out, _ = self.self_attn_1(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        tokens = self.norm_1(tokens + attn_out)

        # Block 2: second self-attention for deeper interaction
        attn_out, _ = self.self_attn_2(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        tokens = self.norm_2(tokens + attn_out)

        class_token_out = tokens[:, 1, :]  # Class token has attended to story + timeline
        class_token_out = class_token_out + class_tok.squeeze(1)  # residual by class embedding
        output = self.ffn(self.head_ln(class_token_out))

        if single:
            output = output.squeeze(0)
        return output
    
    
    def train_model(self, dataloader, epochs=50, learning_rate=5e-4, grad_clip=1.0):
        """
        Train with targets normalized to [-1, 1]. Uses LR scheduler and gradient clipping for stable convergence.
        """
        self.train()
        mse_fn = nn.MSELoss(reduction="mean")
        optimizer = Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_loss_start = 0.0
            epoch_loss_weight = 0.0
            epoch_loss_duration = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                (
                    story_emb,
                    class_emb,
                    timeline_seq,
                    key_padding_mask,
                    targets,
                ) = batch
                story_emb = story_emb.to(self.device)
                class_emb = class_emb.to(self.device)
                timeline_seq = timeline_seq.to(self.device)
                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self(story_emb, class_emb, timeline_seq, key_padding_mask)

                # Single combined loss: per-dimension MSE (one backward is correct)
                loss_start = mse_fn(outputs[:, 0], targets[:, 0])
                loss_weight = mse_fn(outputs[:, 1], targets[:, 1])
                loss_duration = mse_fn(outputs[:, 2], targets[:, 2])
                loss = loss_start + loss_weight + loss_duration

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_start += loss_start.item()
                epoch_loss_weight += loss_weight.item()
                epoch_loss_duration += loss_duration.item()
                num_batches += 1

                if batch_idx == 0 and epoch == 0:
                    logger.info(
                        "[train] First batch sample (normalized): outputs[0]=%s, targets[0]=%s",
                        outputs[0].detach().tolist(),
                        targets[0].tolist(),
                    )

            n = max(num_batches, 1)
            avg_loss = epoch_loss / n
            avg_start = epoch_loss_start / n
            avg_weight = epoch_loss_weight / n
            avg_duration = epoch_loss_duration / n
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | loss=%.4f (start=%.4f, weight_db=%.4f, duration=%.4f) | lr=%.2e | batches=%d",
                epoch + 1, epochs, avg_loss, avg_start, avg_weight, avg_duration, current_lr, num_batches,
            )
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} "
                f"(start: {avg_start:.4f}, weight_db: {avg_weight:.4f}, duration: {avg_duration:.4f}), lr: {current_lr:.2e}"
            )

        return losses

    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self


    def make_whisper_embedding(
        self, story_prompt: str, narrator_audio_base64: str
    ) -> List[dict]:
        """
        Uses a Whisper-based ASR pipeline (via Hugging Face) to generate
        word-level timestamps (whisper_json) from the narrator audio.
        """
        if not hasattr(self, "_word_aligner") or self._word_aligner is None:
            self._word_aligner = WordAligner()

        words_json = self._word_aligner.get_timestamps_from_base64(narrator_audio_base64)
        logger.info(f"Generated Whisper JSON: {words_json}")
        return words_json
    
    
    def get_whisper_sequence_tensor(
        self, words_json: List[dict], scene_duration: float
    ) -> torch.Tensor:
        """
        Converts the entire Whisper JSON into a sequence of temporal-semantic tokens.
        Each token = word_embedding + [start, end, scene_duration]. Shape: [seq_len, embed_dim + 3].
        If no words, returns [1, embed_dim + 3] dummy token.
        """
        if not words_json:
            dummy_emb = torch.zeros(self.embed_dim, device=self.device, dtype=torch.float32)
            dummy_time = torch.tensor(
                [0.0, scene_duration, scene_duration],
                device=self.device,
                dtype=torch.float32,
            )
            return torch.cat([dummy_emb, dummy_time], dim=0).unsqueeze(0)

        word_texts = [w["word"] for w in words_json]
        word_embs = self.embedder.encode(
            word_texts, convert_to_tensor=True, device=str(self.device)
        )
        times = [[w["start"], w["end"], scene_duration] for w in words_json]
        time_tensor = torch.tensor(times, dtype=torch.float32, device=self.device)
        return torch.cat([word_embs, time_tensor], dim=1)

    
    def predict_from_dsp(
        self,
        story_prompt: str,
        audio_classes: List[str],
        whisper_json: List[dict],
        scene_duration: float = 10.0,
    ):
        """
        Takes the story and a list of sounds to generate parameters for.
        Returns a list of [start_time_sec, weight_db, duration_sec] per audio class.
        Uses full Whisper timeline.
        """
        self.eval()
        results: List[Any] = []
        timeline_seq = self.get_whisper_sequence_tensor(whisper_json, scene_duration)
        story_emb = self.embedder.encode(
            story_prompt,
            convert_to_tensor=True,
            device=str(self.device),
        )

        with torch.no_grad():
            for audio_class in audio_classes:
                class_emb = self.embedder.encode(
                    audio_class,
                    convert_to_tensor=True,
                    device=str(self.device),
                )
                output = self(story_emb, class_emb, timeline_seq)  # single cue, returns [3]
                s, w, d = output[0].item(), output[1].item(), output[2].item()
                s_orig, w_orig, d_orig = denormalize_outputs(s, w, d)
                results.append([s_orig, w_orig, d_orig])
                
                logger.info(f"DSP prediction: start_time_sec={s_orig} weight_db={w_orig} duration_sec={d_orig} for audio_class={audio_class}")

        return results



# testing the model
# if __name__ == "__main__":
#     model = CinematicMixPredictor()
#     model.load_model(MODEL_PATH)
#     story_prompt = "A helmet-clad soldier cautiously navigates a grimy, dimly lit urban corridor before being brutally ambushed by a bloodied operative, who then, protecting a young boy, plunges into a chaotic close-quarters gunfight against multiple assailants."
    
#     description = "A male speaker with a neutral tone delivers his words clearly and confidently in a casual, everyday setting."
#     audio_classes = ["Distant urban street ambience", "Tense synth drone with subtle rhythmic percussion", "Heavy tactical footsteps and gear rustle", "Brutal melee combat impacts and vocal grunts", "Pistol slide rack and reload click", "Deep male voice (low dialogue, 'Come on')", "Rapid gunfire, body impacts, and close-quarters combat SFX", "Aggressive percussive action music swell"]
#     narrator_audio_segment = ParlerTTSModel.generate(
#         prompt=story_prompt, description=description
#     )

#     narrator_audio_base64 = audio_to_base64(narrator_audio_segment)
#     results = model.predict(story_prompt, audio_classes, narrator_audio_base64)
#     logger.info(json.dumps(results, indent=2))
