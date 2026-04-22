import sys
import os

# Get absolute path of project root (one level up from current notebook)
project_root = os.path.abspath("..")

# Add to sys.path if not already
if project_root not in sys.path:
    sys.path.append(project_root)

import base64
import io
import json
import logging
import os
import tempfile
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
import laion_clap
from scipy.stats import entropy

from frechet_audio_distance import FrechetAudioDistance
from Variable.dataclases import AudioCue

logger = logging.getLogger(__name__)


YT_JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "yt_videos", "yt_dataset.jsonl"
)


class AudioEvaluator:
    def __init__(self):
        logger.info("Loading CLAP Model for Evaluation...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use default CLAP configuration to match the released checkpoint
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        self.clap_model.to(self.device)

        # Thresholds for story and cue semantic similarity.
        self.THRESHOLD_STORY = 0.7
        self.THRESHOLD_CUE = 0.6

        self.fad_model = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        
        self.yt_stories, self.yt_story_embeds, self.valid_indices = self._load_yt_dataset_and_embeddings()
        
        

    def _base64_to_temp_file(self, audio_base64):
        """Convert base64 audio to a temporary file and return the path"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(audio_bytes)
            return temp_path
        except Exception as e:
            os.close(temp_fd)
            raise e

    def get_audio_richness(self, audio_base64):
        """Measures Spectral Flatness and Entropy (Proxies for quality/complexity) from base64 audio"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes))
        
        # Spectral Flatness: 1.0 = white noise, 0.0 = pure tone. 
        # We want a mid-range for complex scores.
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Spectral Entropy: How 'unpredictable' the sound is.
        S = np.abs(librosa.stft(y))
        psd = np.sum(S**2, axis=1)
        psd /= np.sum(psd)
        spec_entropy = entropy(psd)
        
        return flatness, spec_entropy

    def evaluate_sync_from_audio_base64(self, audio_base64, action_keywords=["shot", "bang", "crash", "door"]):
        """Check if peaks exist in audio (simple onset detection) from base64 audio"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes))
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        # Returns number of sharp transients found
        return len(onsets)
    
    def get_noise_floor(self, audio_base64):
        """Calculate noise floor in dB from base64 audio"""
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_base64:
            audio_base64 = audio_base64.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes))
        
       
        rms = librosa.feature.rms(y=y)[0]
        
        rms_db = librosa.power_to_db(rms, ref=float(np.max(rms)))
        
        noise_floor_db = np.min(rms_db)
        
        return noise_floor_db
    

    def _load_yt_dataset_and_embeddings(self):
        """
        Load the latest YT JSONL and compute text embeddings for each valid story prompt.
        This is called on every evaluation so that we never use stale/cached embeddings,
        and we maintain a correct mapping between embeddings and dataframe rows.
        """
        logger.info("Loading YouTube Ground Truth Dataset from JSONL for this run...")
        yt_records = []
        with open(YT_JSONL_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yt_records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed JSONL line %d in %s: %s",
                        idx,
                        YT_JSONL_PATH,
                        str(e),
                    )
                    continue

        yt_stories = pd.DataFrame(yt_records)

        if "story_prompt" not in yt_stories.columns:
            logger.warning(
                "Column 'story_prompt' not found in YT dataset; "
                "semantic matching will be disabled."
            )
            return yt_stories, None, []

        # Build a list of (row_index, cleaned_text) only for valid non-empty prompts
        valid_indices = []
        valid_texts = []
        for row_idx, raw in yt_stories["story_prompt"].items():
            if isinstance(raw, str):
                cleaned = raw.strip()
                if cleaned:
                    valid_indices.append(row_idx)
                    valid_texts.append(cleaned)

        if not valid_texts:
            logger.warning(
                "No valid non-empty 'story_prompt' texts found in YT dataset; "
                "semantic matching will be disabled."
            )
            return yt_stories, None, []

        logger.info("Embedding %d YT story prompts for retrieval...", len(valid_texts))
        with torch.no_grad():
            yt_story_embeds = self.clap_model.get_text_embedding(
                valid_texts, use_tensor=True
            )

        return yt_stories, yt_story_embeds, valid_indices

    def _find_closest_yt_story(self, generated_story_text: str):
        """Finds the absolute closest matching story in the dataset."""
        # Always reload YT dataset and recompute embeddings so there is no caching
        # across runs and we always reflect the latest contents of yt_dataset.jsonl.
        

        if self.yt_story_embeds is None or len(self.valid_indices) == 0:
            logger.warning(
                "YT story embeddings are not available; skipping semantic retrieval."
            )
            return None

        # First try an exact string match before falling back to embeddings.
        # This lets you effectively treat THRESHOLD_STORY=1.0 as "only accept
        # an exact textual match" for the common case where prompts are identical.
        if "story_prompt" in self.yt_stories.columns:
            exact_matches = self.yt_stories[
                self.yt_stories["story_prompt"].astype(str) == str(generated_story_text)
            ]
            if not exact_matches.empty:
                logger.info("Exact textual match found in YT dataset for story prompt.")
                return exact_matches.iloc[0]

        with torch.no_grad():
            gen_embed = self.clap_model.get_text_embedding([generated_story_text], use_tensor=True)
            
            similarities = torch.nn.functional.cosine_similarity(gen_embed, self.yt_story_embeds)
            
            # logger.info(f"\n\nSimilarities: {similarities}\n\n")
            
            max_sim_value, max_sim_index = torch.max(similarities, dim=0)
            # logger.info(f"Max Similarity Value: {max_sim_value.item()}")
            # logger.info(f"Max Similarity Index: {max_sim_index.item()}")
            
            if max_sim_value.item() >= self.THRESHOLD_STORY:
                # Map from embedding index back to the original dataframe row index.
                yt_row_index = self.valid_indices[int(max_sim_index.item())]
                closest_story = self.yt_stories.loc[yt_row_index]
                return closest_story
            return None

    def yt_coverage_score(self, generated_story_text: str, generated_cues: List[AudioCue]):
        """Get YT sync score from story text and audio cues"""
        closest_story = self._find_closest_yt_story(generated_story_text)
        if closest_story is None:
            return {"error": "No semantic match found in YT dataset to establish ground truth."}
            
        # Parse the ground truth cues (assuming they are stored as JSON/dicts in the CSV)
        import ast
        try:
            yt_cues = ast.literal_eval(closest_story["cues"])
        except:
            yt_cues = closest_story["cues"] # If already a list of dicts

        with torch.no_grad():
            gen_cue_texts = [cue.audio_class for cue in generated_cues]
            yt_cue_texts = [cue["audio_class"] for cue in yt_cues]
            
            gen_cue_embeds = self.clap_model.get_text_embedding(gen_cue_texts, use_tensor=True)
            yt_cue_embeds = self.clap_model.get_text_embedding(yt_cue_texts, use_tensor=True)

        if not yt_cues:
            return 0.0

        matched_yt_indices = set()
    
        for i, gen_embed in enumerate(gen_cue_embeds):
            sims = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), yt_cue_embeds)
            best_match_val, best_match_idx = torch.max(sims, dim=0)
            
            if best_match_val.item() >= self.THRESHOLD_CUE:
                matched_yt_indices.add(best_match_idx.item())

        coverage_score_percent = (len(matched_yt_indices) / max(len(yt_cues), 1)) * 100.0
        # Avoid any floating point overshoot
        return float(min(coverage_score_percent, 100.0))
    
    def yt_sync_score(self, generated_story_text: str, generated_cues: List[AudioCue],total_duration_sec: float=16.0):
        """Get YT sync score from story text and audio cues"""
        closest_story = self._find_closest_yt_story(generated_story_text)
        if closest_story is None:
            return {"error": "No semantic match found in YT dataset to establish ground truth."}
        
        logger.info(f"\n\nClosest YT story: {closest_story}\n\n")
        # Parse the ground truth cues (assuming they are stored as JSON/dicts in the CSV)
        import ast
        try:
            yt_cues = ast.literal_eval(closest_story["cues"])
        except:
            yt_cues = closest_story["cues"] # If already a list of dicts

        # 1. Embed all cue descriptions for matching
        with torch.no_grad():
            gen_cue_texts = [cue.audio_class for cue in generated_cues]
            yt_cue_texts = [cue["audio_class"] for cue in yt_cues]
            
            gen_cue_embeds = self.clap_model.get_text_embedding(gen_cue_texts, use_tensor=True)
            yt_cue_embeds = self.clap_model.get_text_embedding(yt_cue_texts, use_tensor=True)

        if not yt_cues or not generated_cues:
            return 0.0

        aligned_pairs = []
        
        # Find which generated cues match which YT cues
        for i, gen_embed in enumerate(gen_cue_embeds):
            sims = torch.nn.functional.cosine_similarity(gen_embed.unsqueeze(0), yt_cue_embeds)
            best_match_val, best_match_idx = torch.max(sims, dim=0)
            
            # preprocess the best match value to fit in scale of total duration of the story
            
            
            if best_match_val.item() >= self.THRESHOLD_CUE:
                
                
                
                best_match_audio_cue = yt_cues[best_match_idx.item()]
                best_match_audio_cue_start_sec = best_match_audio_cue.get("starting_time", 0.0) / total_duration_sec * best_match_audio_cue.get("clip_duration_sec", 1.0)
                best_match_audio_cue_duration_sec = best_match_audio_cue.get("duration", 0.0) / total_duration_sec * best_match_audio_cue.get("clip_duration_sec", 1.0)
                
                best_match_audio_cue["starting_time"] = best_match_audio_cue_start_sec
                best_match_audio_cue["duration"] = best_match_audio_cue_duration_sec
                
                
                logger.info(f"best_match_audio_cue: {best_match_audio_cue} for generated cue: {generated_cues[i]}")
                
                
                aligned_pairs.append({
                    "gen": generated_cues[i],
                    "yt": best_match_audio_cue
                })

        if not aligned_pairs:
            return 0.0

        # Calculate per-cue interval IOU in milliseconds.
        def _interval_ms_from_audio_cue(cue: AudioCue) -> tuple[float, float]:
            start_ms = float(cue.start_time_ms)
            end_ms = start_ms + float(cue.duration_ms)
            return start_ms, end_ms

        def _interval_ms_from_yt_cue(yt_cue: dict) -> tuple[float, float]:
            # YT dataset stores cue times in seconds.
            start_s = float(yt_cue.get("starting_time", 0.0))
            dur_s = float(yt_cue.get("duration", 0.0))
            start_ms = start_s * 1000.0
            end_ms = start_ms + dur_s * 1000.0
            return start_ms, end_ms

        iou_sum = 0.0
        for pair in aligned_pairs:
            gen = pair["gen"]
            yt = pair["yt"]

            gen_start, gen_end = _interval_ms_from_audio_cue(gen)
            yt_start, yt_end = _interval_ms_from_yt_cue(yt)

            inter_start = max(gen_start, yt_start)
            inter_end = min(gen_end, yt_end)
            intersection = max(0.0, inter_end - inter_start)

            union_start = min(gen_start, yt_start)
            union_end = max(gen_end, yt_end)
            union = max(0.0, union_end - union_start)
            logger.info(f"intersect start: {inter_start}, intersect end: {inter_end}, intersection: {intersection}, union start: {union_start}, union end: {union_end}, union: {union}")

            iou = (intersection / union) if union > 0 else 0.0
            iou_sum += iou
        logger.info(f"iou_sum: {iou_sum}, aligned_pairs: {len(aligned_pairs)}")
        return float(iou_sum / len(aligned_pairs))
    
    def yt_coverage_and_sync_score(self, generated_story_text: str, generated_cues: List[AudioCue], total_duration_sec: float=16.0):
        """Get YT coverage and sync score from story text and audio cues"""
        coverage_score = self.yt_coverage_score(generated_story_text, generated_cues)
        sync_score = self.yt_sync_score(generated_story_text, generated_cues, total_duration_sec)
        return {
            "coverage_score": coverage_score,
            "sync_score": sync_score
        }
        

    def get_clap_score(self, audio_base64: str, text_prompt: str) -> float:
        """Measures global Text-to-Audio Alignment."""
        temp_path = self._base64_to_temp_file(audio_base64)
        try:
            audio_embed = self.clap_model.get_audio_embedding_from_filelist(x=[temp_path], use_tensor=True)
            text_embed = self.clap_model.get_text_embedding([text_prompt], use_tensor=True)
            return torch.nn.functional.cosine_similarity(audio_embed, text_embed).item()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_cinematic_acoustic_metrics(self, audio_base64: str) -> Dict[str, float]:
        """Calculates Spectral Richness and Dynamic Range for cinematic quality."""
        temp_path = self._base64_to_temp_file(audio_base64)
        try:
            y, sr = librosa.load(temp_path)
            if len(y) == 0:
                return {}

            # Dynamic Range & Crest Factor (Punchiness)
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            crest_factor = peak / (rms + 1e-10)
            
            noise_floor = np.min(librosa.feature.rms(y=y)[0])
            dynamic_range_db = 20 * np.log10(peak / (noise_floor + 1e-10))

            # Spectral Metrics
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            S = np.abs(librosa.stft(y))
            psd = np.sum(S**2, axis=1)
            psd /= (np.sum(psd) + 1e-10)
            spec_entropy = entropy(psd)

            return {
                "dynamic_range_db": float(dynamic_range_db),
                "crest_factor": float(crest_factor),
                "spectral_flatness": float(flatness),
                "spectral_entropy": float(spec_entropy),
                "spectral_centroid_hz": float(centroid),
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_spectral_kl_divergence(self, gen_audio_base64: str, ref_audio_path: str) -> float:
        """Calculates how closely the generated frequency distribution matches the real audio."""
        temp_path = self._base64_to_temp_file(gen_audio_base64)
        try:
            y_gen, _ = librosa.load(temp_path, sr=16000)
            y_ref, _ = librosa.load(ref_audio_path, sr=16000)
            
            def get_psd(y):
                S = np.abs(librosa.stft(y))
                psd = np.sum(S**2, axis=1)
                return psd / (np.sum(psd) + 1e-10)
                
            psd_gen = get_psd(y_gen)
            psd_ref = get_psd(y_ref)
            
            # KL(Ref || Gen)
            return float(entropy(psd_ref, psd_gen))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
   
    def calculate_fad_score(self, generated_audio_dir: str):
        """
        Calculates FAD and Inception Score over an entire folder of generated audio.
        Requires directories containing standard .wav files.
        """
        logger.info("Calculating Fréchet Audio Distance (FAD)...")
        if self.fad_model is None:
            raise RuntimeError(
                "fad_model is not initialized. Wire up a FAD implementation before calling calculate_fad_score()."
            )
        # Calculates statistical distance between real audio distribution and generated audio distribution
        ground_truth_audio_dir = os.path.join(os.path.dirname(__file__), "..", "data", "yt_videos", "yt_dataset")
        fad_score = self.fad_model.score(ground_truth_audio_dir, generated_audio_dir)
        logger.info("Calculating Inception Score (IS)...")
        return fad_score     

if __name__ == "__main__":
    evaluator = AudioEvaluator()
    # evaluator.evaluate_sync("generated_score.wav")
    # c_score = evaluator.get_clap_score("generated_score.wav", "i followed a dog where i heard a gunshot and footsteps apporaching me")
    # flat, ent = evaluator.get_audio_richness("generated_score.wav")
    # peak_count = evaluator.evaluate_sync("generated_score.wav")
    # print(f"""
    # --- Evaluation Results (No Ground Truth) ---
    # Text-Audio Alignment (CLAP): {c_score:.4f}  (Target: >0.25 for good match)
    # Spectral Entropy (Richness): {ent:.4f}      (Target: Higher = more complex music)
    # Spectral Flatness (Noise):   {flat:.4f}     (Target: Lower = more tonal/musical)
    # Detected Audio Onsets:       {peak_count}   (Number of dynamic events)
    # """)
    
    # {"audio_class": "male character gasps and strains", "starting_time": 0.5, "duration": 9.0, "weight_db": -10.0}, {"audio_class": "intense water splashing SFX", "starting_time": 0.0, "duration": 10.0, "weight_db": -12.0}]
    
 
  
    story_prompt = "The clip rapidly transitions from serene futuristic grandeur and initial technological hope to widespread catastrophic destruction, revealing the gritty aftermath of a devastated world, followed by escalating human and robotic conflict across varied landscapes, culminating in a glimpse of a mysterious, possibly critical, futuristic stronghold amidst snowy peaks."
    cues = [AudioCue(id=0, audio_class="Deep male narrator (expository)", audio_type="SFX", start_time_ms=0, duration_ms=6000, weight_db=-10.0), AudioCue(id=1, audio_class="Catastrophic nuclear explosion and rumble SFX", audio_type="SFX", start_time_ms=2000, duration_ms=2500, weight_db=-5.0), AudioCue(id=2, audio_class="Ethereal female choir (mournful tone)", audio_type="SFX", start_time_ms=6000, duration_ms=3000, weight_db=-20.0), AudioCue(id=3, audio_class="Urgent male narrator (existential conflict)", audio_type="SFX", start_time_ms=8800, duration_ms=2000, weight_db=-10.0), AudioCue(id=4, audio_class="Orchestral action score with tense build-up and driving percussion", audio_type="SFX", start_time_ms=9000, duration_ms=11000, weight_db=-12.0), AudioCue(id=5, audio_class="Hard cinematic impact and whoosh SFX", audio_type="SFX", start_time_ms=11000, duration_ms=500, weight_db=-8.0), AudioCue(id=6, audio_class="Military male narrator (strategic briefing)", audio_type="SFX", start_time_ms=14000, duration_ms=2000, weight_db=-10.0), AudioCue(id=7, audio_class="Explosions, futuristic gunfire, and laser blasts SFX", audio_type="SFX", start_time_ms=13500, duration_ms=5000, weight_db=-7.0), AudioCue(id=8, audio_class="Military male narrator (revealing new threat, cut off)", audio_type="SFX", start_time_ms=18500, duration_ms=1500, weight_db=-10.0)]
    print(f"Story Prompt: {story_prompt}")
    print(f"Cues: {cues}\n\n")
    
    total_duration_ms = max(cue.start_time_ms + cue.duration_ms for cue in cues)
    total_duration_sec = total_duration_ms / 1000.0

    coverage_and_sync_score = evaluator.yt_coverage_and_sync_score(story_prompt, cues, total_duration_sec)
    # print(f"Clap Score: {evaluator.get_clap_score(audio_base64, text_prompt)}")
    # print(f"Spectral Richness: {evaluator.get_spectral_richness(audio_base64)}")
    # print(f"Noise Floor: {evaluator.get_noise_floor(audio_base64)}")
    # print(f"Audio Onsets: {evaluator.evaluate_sync(audio_base64)}")
    # print(f"Coverage and Sync Score: {coverage_and_sync_score}")
    # print(f"FAD Score: {evaluator.calculate_fad_score(audio_base64)}")
    # print(f"Spectral KL Divergence: {evaluator.get_spectral_kl_divergence(audio_base64, ref_audio_path)}")
    # print(f"Cinematic Acoustic Metrics: {evaluator.get_cinematic_acoustic_metrics(audio_base64)}")
