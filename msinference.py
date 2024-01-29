import time
import random
from collections import OrderedDict

import yaml
import nltk
import torch
import librosa
import numpy as np
import torchaudio
import phonemizer
from torch import nn
import torch.nn.functional as F
from munch import Munch
from pydub import AudioSegment
from nltk.tokenize import word_tokenize
from src.utils.audio import save_wav

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

textclenaer = TextCleaner()
nltk.download("punkt")


def load_model(config_path, ckpt_path):
    config = yaml.safe_load(open(config_path))

    # Load pretrained ASR model, F0 and BERT models
    plbert = load_plbert(config.get('PLBERT_dir', False))
    pitch_extractor = load_F0_models(config.get("F0_path", False))
    text_aligner = load_ASR_models(config.get("ASR_path", False), config.get("ASR_config",False))

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to("cuda") for key in model]

    params_whole = torch.load(ckpt_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    _ = [model[key].eval() for key in model]
    return model, model_params

class TTSPredictor():
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch="ignore")
        self.model_ref, self.model_ref_config = load_model(config_path="checkpoints/LibriTTS/config.yml", ckpt_path="checkpoints/LibriTTS/epochs_2nd_00020.pth")
        self.sampler_ref = DiffusionSampler(
            self.model_ref.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4

        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model_ref.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model_ref.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)


    def inference_with_ref(
        self, text, ref_s, s_prev=None, alpha=0.3, beta=0.7, t=0.7, 
        diffusion_steps=5, embedding_scale=1, trim=50, longform=False
    ):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        ps = ps.replace('``', '"')
        ps = ps.replace("''", '"')

        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model_ref.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model_ref.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model_ref.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler_ref(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, 
                num_steps=diffusion_steps
            ).squeeze(1)

            # convex combination of previous and current style
            if s_prev is not None:
                s_pred = t * s_prev + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            if s_prev is not None or longform==True:
                s_pred = torch.cat([ref, s], dim=-1)

            d = self.model_ref.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model_ref.predictor.lstm(d)
            duration = self.model_ref.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_ref_config.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model_ref.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_ref_config.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model_ref.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
        return out.squeeze().cpu().numpy()[..., :-trim], s_pred
    
    def predict(
        self,
        text: str,
        reference: str = 'examples/audio_style/m-us-3.wav',
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 1,
        embedding_scale: float = 1,
        seed: int = 0
    ) -> str:
        """Run a single prediction on the model"""

        print("Running TTS predictions...")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if reference is None:
            print("No reference audio provided.")
        else:
            ref_s = self.compute_style(str(reference))
            if len(text) >= 400:
                wavs = []
                s_prev = None
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    if sent.strip() == "": continue
                    wav, s_prev = self.inference_with_ref(
                        sent, ref_s, s_prev, alpha=alpha, beta=beta, t=0.7, trim=100,
                        diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, longform=True 
                    )
                    wavs.append(wav)
                wav = np.concatenate(wavs)
            else:
                noise = torch.randn(1, 1, 256).to(self.device)
                ref_s = self.compute_style(str(reference))
                wav, _ = self.inference_with_ref(
                    text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale
                )

        out_path = "/tmp/audio_out.wav"
        save_wav(wav, out_path, sr=24000)
        return out_path
