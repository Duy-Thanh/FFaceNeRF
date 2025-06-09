# FFaceNeRF
### [CVPR2025] FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields

### [[Project Page](https://kwanyun.github.io/FFaceNeRF_page/)] [[Paper](https://arxiv.org/abs/2503.08417)]

![teaser](https://github.com/user-attachments/assets/b51980f8-29ae-46ec-a572-6700ae0462ae)

### Be aware current version still need some correction and clean-ups. If there are any suggestion for environment setting, let us know for future users



## :gear: Install Environment via Anaconda (Recommended)
    conda env create -f environment.yml
    conda activate ffacenerf


FFaceNeRF requires NeRRFaceEditing checkpoints for intialization

put [pretrained_model](https://drive.google.com/file/d/1N4y3leKEF7rbMVNbpYUYtNnaO4WVDln1/view?usp=drive_link) into networks/NeRFFaceEditing-ffhq-64.pkl

Download [Data](https://drive.google.com/file/d/16ha-UeU2uLZu7YNYPXw-I1yIHyav2E0O/view?usp=drive_link) for training and testing

python train_ffacenerf.py --mode eyes



python editing_testset.py --mode eyes --network ckpt_eyes_10

### In the original CVPR paper, the test set comprised 22 samples; in this public repository, we expanded it to 41 for the testing.

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
