import json
import os
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from tango_new.tango2.models import AudioDiffusion, DDPMScheduler
from tango_new.tango2.audioldm.audio.stft import TacotronSTFT
from tango_new.tango2.audioldm.variational_autoencoder import AutoencoderKL

def _get_device(device=None):
    """Resolve device: use CUDA if available, else MPS (Apple Silicon), else CPU."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Tango:
    def __init__(self, name="declare-lab/tango", device=None):
        device = _get_device(device)

        path = snapshot_download(repo_id=name)
        
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))

        # Resolve relative paths in main_config to the downloaded repo root (diffusers expects local paths).
        # declare-lab/tango2 on HF does not include configs/; use bundled config from this package if missing.
        if main_config.get("unet_model_config_path"):
            p = main_config["unet_model_config_path"].strip()
            if not os.path.isabs(p):
                p = os.path.normpath(os.path.join(path, p))
            if not os.path.isfile(p):
                _dir = os.path.dirname(os.path.abspath(__file__))
                fallback = os.path.join(_dir, "tango2", "configs", "diffusion_model_config.json")
                if os.path.isfile(fallback):
                    p = fallback
            main_config["unet_model_config_path"] = p

        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = AudioDiffusion(**main_config).to(device)
        
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=device)
        
        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print ("Successfully loaded checkpoint from:", name)
        
        self.vae.eval()
        self.stft.eval()
        self.model.eval()
        
        self.scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")
        
    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """ Genrate audio for a single prompt string. """
        with torch.no_grad():
            latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]
    
    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """ Genrate audio for a list of prompt strings. """
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k: k+batch_size]
            with torch.no_grad():
                latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
                mel = self.vae.decode_first_stage(latents)
                wave = self.vae.decode_to_waveform(mel)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))