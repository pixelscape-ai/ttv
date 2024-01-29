# SadTalker-Video
[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gauravk95/SadTalker-Video/blob/master/sad_talker_video_colab.ipynb) &nbsp; [![Replicate](https://replicate.com/gauravk95/sadtalker-video/badge)](https://replicate.com/gauravk95/sadtalker-video)

This project is based on SadTalkers to implement Wav2lip for video lip synthesis. By using video files to generate lip shapes driven by voice, and setting a configurable enhancement method for the facial area, the synthetic lip shape (face) area image enhancement is performed to improve the clarity of the generated lip shapes. Use the DAIN frame interpolation DL algorithm to add frames to the generated video to supplement the action transition of synthetic lip shapes between frames, making the synthesized lip shapes more smooth, realistic and natural.

## 1. Environment preparation (Environment)

```python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install ffmpeg
pip install -r requirements.txt

#If you need to use the DAIN model for frame filling, you need to install paddle.
# CUDA 11.2
python -m pip install paddlepaddle-gpu==2.3.2.post112 \
-f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## 2. Project structure (Repository structure)

```
SadTalker-Video-Lip-Sync
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├── ...
├──dian_output
|   ├── ...
├──examples
|   ├── audio
|   ├── video
├──results
|   ├── ...
├──src
|   ├── ...
├──sync_show
├──third_part
|   ├── ...
├──...
├──inference.py
├──README.md
```

## 3. Model inference (Inference)

```python
python inference.py --driven_audio <audio.wav> \
                    --source_video <video.mp4> \
                    --enhancer <none,lip,face> \  #(default lip)
                    --use_DAIN \ #(Using this function will occupy a large amount of video memory and consume more time)
             		--time_step 0.5 #(Frame insertion frequency, default 0.5, that is, 25fps—>50fps; 0.25, that is, 25fps—>100fps)
```



## 4.Synthetic effects (Results)

```python
#The synthesis effect is displayed in the ./sync_show directory:
#original.mp4 Original video
#sync_none.mp4 No enhanced synthesis effects
#none_dain_50fps.mp4 Add frames from 25fps to 50fps using DAIN model only
#lip_dain_50fps.mp4 Enhance the lip area to make the lip shape clearer + DAIN model adds frames from 25fps to 50fps
#face_dain_50fps.mp4 Enhance the entire face area to make the lip shape clearer + DAIN model adds frames from 25fps to 50fps

#The following is a video of the generation effects of different methods
#our.mp4 Video generated by SadTalker-Video-Lip-Sync in this project
#sadtalker.mp4 full video generated by sadtalker
#retalking.mp4 Video generated by retalking
#wav2lip.mp4 Video generated by wav2lip
```

https://user-images.githubusercontent.com/52994134/231769817-8196ef1b-c341-41fa-9b6b-63ad0daf14ce.mp4

When the videos are spliced together, the frame number is unified to 25fps. The effect of interpolating frames cannot be seen. For specific details, you can see the individual videos in the ./sync_show directory for comparison.

**Comparison of the effects of this project with sadtalker, retalking, and wav2lip lip synthesis:**

|                           **our**                            |                        **sadtalker**                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <video  src="https://user-images.githubusercontent.com/52994134/233003969-91fa9e94-a958-4e2d-b958-902cc7711b8a.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003985-86d0f75c-d27f-4a52-ac31-2649ccd39616.mp4" type="video/mp4"> </video> |
|                        **retalking**                         |                         **wav2lip**                          |
| <video  src="https://user-images.githubusercontent.com/52994134/233003982-2fe1b33c-b455-4afc-ab50-f6b40070e2ca.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003990-2f8c4b84-dc74-4dc5-9dad-a8285e728ecb.mp4" type="video/mp4"> </video> |

The video displayed in the readme has been resized. The original video can be compared by viewing the synthesized videos of different categories in the ./sync_show directory.

## 5. Pretrained model (Pretrained model)

The pretrained model looks like this:

```python
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├──auido2exp_00300-model.pth
|   ├──auido2pose_00140-model.pth
|   ├──epoch_20.pth
|   ├──facevid2vid_00189-model.pth.tar
|   ├──GFPGANv1.3.pth
|   ├──GPEN-BFR-512.pth
|   ├──mapping_00109-model.pth.tar
|   ├──ParseNet-latest.pth
|   ├──RetinaFace-R50.pth
|   ├──shape_predictor_68_face_landmarks.dat
|   ├──wav2lip.pth
```

Pre-trained model checkpoints download path:

Baidu Netdisk: https://pan.baidu.com/s/15-zjk64SGQnRT9qIduTe2A Extraction code: klfv

Google Cloud Drive: https://drive.google.com/file/d/1lW4mf5YNtS4MAD7ZkAauDDWp2N3_Qzs7/view?usp=sharing

Quark network disk: https://pan.quark.cn/s/2a1042b1d046 Extraction code: zMBP

```python
#Download the compressed package and extract it to the project path (need to be executed when downloading Google Cloud Disk and Quark Cloud Disk)
cd SadTalker-Video-Lip-Sync
tar -zxvf checkpoints.tar.gz
```

## Reference
- SadTalker:https://github.com/Winfredy/SadTalker
-  VideoReTalking：https://github.com/vinthony/video-retalking
- DAIN :https://arxiv.org/abs/1904.00830
- PaddleGAN:https://github.com/PaddlePaddle/PaddleGAN