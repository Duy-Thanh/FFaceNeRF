# FFaceNeRF
### [CVPR2025] FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields

### [[Project Page](https://kwanyun.github.io/FFaceNeRF_page/)] [[Paper](https://arxiv.org/abs/2503.08417)]

![teaser](https://github.com/user-attachments/assets/b51980f8-29ae-46ec-a572-6700ae0462ae)

### Be aware current version still need some correction and clean-ups. If there are any suggestion for environment setting, let us know for future users



## :gear: Install Environment via Anaconda3 (version 2024.10) - (Required)
    conda env create -f environment.yml
    conda activate ffacenerf


FFaceNeRF requires NeRRFaceEditing checkpoints for intialization

This program fine-tuned to run on NVIDIA GeForce RTX 3060 (8GB VRAM)

You **must** have NVIDIA GeForce RTX 3060 (8GB VRAM) or better, and you **must** have at least 16 GB RAM

You must install these programs **BEFORE** you can continue: **Anaconda3 (version 2024.10), Microsoft Visual Studio 2019 Build Tools (C++ workloads only), NVIDIA GPU Computing Toolkit CUDA 11.3**

You can find them in here:

- NVIDIA GPU Computing Toolkit CUDA 11.3: [https://developer.download.nvidia.com/compute/cuda/11.3.0/network_installers/cuda_11.3.0_win10_network.exe](https://developer.download.nvidia.com/compute/cuda/11.3.0/network_installers/cuda_11.3.0_win10_network.exe)
- Microsoft Visual Studio 2019 Build Tools (only C++ Workloads required): [https://aka.ms/vs/16/release/vs_buildtools.exe](https://aka.ms/vs/16/release/vs_buildtools.exe)
- Anaconda3: [https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe)

## Gathering Files

Run these step **ONLY AFTER** you completed **:gear: Install Environment via Anaconda3 (version 2024.10) - (Required)**

- put [pretrained_model](https://drive.google.com/file/d/1N4y3leKEF7rbMVNbpYUYtNnaO4WVDln1/view?usp=drive_link) into networks/NeRFFaceEditing-ffhq-64.pkl

- Download [Data](https://drive.google.com/file/d/16ha-UeU2uLZu7YNYPXw-I1yIHyav2E0O/view?usp=drive_link) for training and testing

- Download [this](https://github.com/Duy-Thanh/FFaceNeRF/releases/download/release/networks.7z) in **root directory of the project where you have cloned before (for example: D:\FFaceNeRF)** and **select extract here**

- Download [this](https://github.com/Duy-Thanh/FFaceNeRF/releases/download/release/vgg16-397923af.7z) to `%USERPROFILE%\.cache\torch\hub\checkpoints` **(Create if not exists)** then **select extract here**

- Download [this](https://github.com/Duy-Thanh/FFaceNeRF/blob/main/freeimage.zip) to `<PATH_WHERE_YOUR_ANNACONDA3_INSTALLED>\envs\ffacenerf\Lib\site-packages\imageio\resources` then **select extract here**

## Training requires about 40 minutes on single A6000 GPU (you can do it now on RTX 3060 with only 8GB VRAM)
    python train_ffacenerf.py --mode eyes
    #python train_ffacenerf.py --mode nose
    #python train_ffacenerf.py --mode chin


## Testing
    python editing_testset.py --mode eyes --network ckpt_eyes_10.pth --overlap_weight 0.5
    #python editing_testset.py --mode eyes --network ckpt_nose_10.pth --overlap_weight 0.4 --target_image 64



In the original CVPR paper, the test set comprised 22 samples; in this public repository, we expanded it to 41 for the testing.

    python evaluate.py

#### We would like to thank EG3D and NeRFFaceEditing for open-source video interpolation model

## Citation
```
@inproceedings{yun2025ffacenerf,
  title={FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields},
  author={Yun, Kwan and Kim, Chaelin and Shin, Hangyeul and Noh, Junyong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10825--10835},
  year={2025}
}
```
