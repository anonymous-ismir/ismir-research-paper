import json
import os
import sys
import torch
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from tango_new.tango2.models import AudioDiffusion, DDPMScheduler
from tango_new.tango2.audioldm.audio.stft import TacotronSTFT
from tango_new.tango2.audioldm.variational_autoencoder import AutoencoderKL
import logging

logger = logging.getLogger(__name__)

def _get_device(device=None):
    """Use CUDA if available, else MPS (Apple Silicon), else CPU."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class Tango:
    def __init__(self, name="declare-lab/tango", device=None):
        device = _get_device(device)
        
        path = snapshot_download(repo_id=name)
        logger.info(f"Downloaded checkpoint from: {path}")
        
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))

        # Resolve relative UNet config path against the snapshot dir (avoids HF 404)
        if main_config.get("unet_model_config_path"):
            cfg_path = main_config["unet_model_config_path"]
            if not os.path.isabs(cfg_path):
                main_config = dict(main_config)
                main_config["unet_model_config_path"] = os.path.normpath(os.path.join(path, cfg_path))

        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = AudioDiffusion(**main_config).to(device)
        
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=device)
        
        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print("Successfully loaded checkpoint from:", name)
        print("Using device:", device)
        
        self.vae.eval()
        self.stft.eval()
        self.model.eval()
        
        self.scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")
        
    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=False, duration=10.0):
        """Generate audio for a single prompt string.

        Args:
            prompt: Text description of the audio to generate.
            steps: Number of diffusion steps.
            guidance: Classifier-free guidance scale.
            samples: Number of samples per prompt.
            disable_progress: Whether to hide the progress bar.
            duration: Target duration of the generated audio in seconds (e.g. 3.0, 5.0, 10.0).
        """
        with torch.no_grad():
            latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress, duration=duration)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]

    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=False, duration=10.0):
        """Generate audio for a list of prompt strings.

        Args:
            prompts: List of text descriptions.
            steps: Number of diffusion steps.
            guidance: Classifier-free guidance scale.
            samples: Number of samples per prompt.
            batch_size: Batch size for generation.
            disable_progress: Whether to hide the progress bar.
            duration: Target duration of each generated audio in seconds (e.g. 3.0, 5.0, 10.0).
        """
        outputs = []
        batch_iter = range(0, len(prompts), batch_size)
        pbar = tqdm(
            batch_iter,
            desc="Tango batches",
            disable=disable_progress,
            file=sys.stderr,
            mininterval=0.5,
            leave=True,
        )
        for k in pbar:
            batch = prompts[k: k+batch_size]
            with torch.no_grad():
                latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress, duration=duration)
                mel = self.vae.decode_first_stage(latents)
                wave = self.vae.decode_to_waveform(mel)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))