"""run bash download.sh first to prepare the weights file"""
import torch
from time import strftime
import os, sys
import shutil
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from third_part.GFPGAN.gfpgan import GFPGANer
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from cog import BasePredictor, Input, Path
import warnings
from scipy.io.wavfile import write
from msinference import TTSPredictor

warnings.filterwarnings("ignore")

checkpoint_dir = "checkpoints"
result_dir = "results"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setup Predictions...")
        device = "cuda"

        # Text to Audio
        self.tts = TTSPredictor()
        self.tts.setup()
        
        # Audio2VID MODELS
        path_of_lm_croper = os.path.join(checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
        path_of_net_recon_model = os.path.join(checkpoint_dir, 'epoch_20.pth')
        dir_of_BFM_fitting = os.path.join(checkpoint_dir, 'BFM_Fitting')
        wav2lip_checkpoint = os.path.join(checkpoint_dir, 'wav2lip.pth')

        audio2pose_checkpoint = os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth')
        audio2pose_yaml_path = os.path.join('src', 'config', 'auido2pose.yaml')

        audio2exp_checkpoint = os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth')
        audio2exp_yaml_path = os.path.join('src', 'config', 'auido2exp.yaml')

        free_view_checkpoint = os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar')

        mapping_checkpoint = os.path.join(checkpoint_dir, 'mapping_00109-model.pth.tar')
        facerender_yaml_path = os.path.join('src', 'config', 'facerender_still.yaml')

        # init model
        print("Loading Model Weights...")
        self.preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

        self.audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, audio2exp_checkpoint, audio2exp_yaml_path,
                                    wav2lip_checkpoint, device)
        self.animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, device)

        self.restorer_model = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean',
                                channel_multiplier=2, bg_upsampler=None)
        self.enhancer_model = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False,
                                        sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
        print("All Models Loaded :)")
    

    def predict(
        self,
        video_input_path: Path = Input(
            description="Upload the source video usually a .mp4 file",
        ),
        audio_input_path: Path = Input(
            description="Upload the driven audio, accepts .wav and .mp4 file",
        ),
        text: str = Input(
            description="Text",
            default=""
        ),
        enhancer_region: str = Input(
            description="Choose a face enhancer region",
            choices=["none", "lip", "face"],
            default="lip",
        ),
        use_DAIN: bool = Input(
            description="Enable Depth-Aware Video Frame Interpolation",
            default=False,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print("Start Predictions...")

            # set default configs
        device = "cuda"
        batch_size = 1
        dain_output = 'dain_output'
        dain_weight = 'checkpoints/DAIN_weight'
        dain_time_step = 0.5
        dain_remove_duplicates = False
        
        video_path = str(video_input_path)
        audio_path = str(audio_input_path)

        # if input is text, use the tts and use input audio as ref
        if text:
            new_audio_path = self.tts.predict(text=text, reference=audio_path, alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1)
            # update the audio
            audio_path = new_audio_path

        
        
        # set basic peridictions params
        save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)
    
        # process video
        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(video_path, first_frame_dir)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return
        # audio2ceoff
        batch = get_data(first_coeff_path, audio_path, device)
        coeff_path = self.audio_to_coeff.generate(batch, save_dir)
        # coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, device)
        tmp_path, new_audio_path, return_path = self.animate_from_coeff.generate(data, save_dir, video_path, crop_info, self.restorer_model, self.enhancer_model, enhancer_region)
        final_mp4_path = return_path 
        
        torch.cuda.empty_cache()
        if use_DAIN:
            import paddle
            from src.dain_model import dain_predictor
            paddle.enable_static()
            predictor_dain = dain_predictor.DAINPredictor(dain_output, weight_path = dain_weight,
                                                        time_step = dain_time_step,
                                                        remove_duplicates = dain_remove_duplicates)
            frames_path, temp_video_path = predictor_dain.run(tmp_path)
            paddle.disable_static()
            save_path = return_path[:-4] + '_dain.mp4'
            final_mp4_path = save_path
            command = r'ffmpeg -y -i "%s" -i "%s" -c:v libx264 "%s"' % (temp_video_path, new_audio_path, save_path)
            os.system(command)

        # send the final output
        output = "/tmp/out.mp4"
        shutil.copy(final_mp4_path, output)

        os.remove(tmp_path)
        return Path(output)
