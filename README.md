# TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

## [Project page](https://patrickddj.github.io/TimeFormer/) | [Paper](https://arxiv.org/abs/2411.11941)
![image](https://patrickddj.github.io/TimeFormer/images/timeformer.png)

Dynamic scene reconstruction is a long-term challenge in 3D vision. Recent methods extend 3D Gaussian Splatting to dynamic scenes via additional deformation fields and apply explicit constraints like motion flow to guide the deformation. However, they learn motion changes from individual timestamps independently, making it challenging to reconstruct complex scenes, particularly when dealing with violent movement, extreme-shaped geometries, or reflective surfaces. To address the above issue, we design a plug-and-play module called TimeFormer to enable existing deformable 3D Gaussians reconstruction methods with the ability to implicitly model motion patterns from a learning perspective. Specifically, TimeFormer includes a Cross-Temporal Transformer Encoder, which adaptively learns the temporal relationships of deformable 3D Gaussians. Furthermore, we propose a two-stream optimization strategy that transfers the motion knowledge learned from TimeFormer to the base stream during the training phase. This allows us to remove TimeFormer during inference, thereby preserving the original rendering speed. Extensive experiments in the multi-view and monocular dynamic scenes validate qualitative and quantitative improvement brought by TimeFormer.



> Note: TimeFormer can be easily **plug-and-play to all existing and future deformable 3D Gaussian framework!**
>
> All you need include:
>
> 	1. Copy TimeFormer.py
> 	1. Add TimeFormer branch and its loss
> 	1. Add backpropagation of TimeFormer





## Environment
```shell
pip install torch torchvision
pip install -r requirements.txt
```


## Dataset
```shell
mkdir data
mkdir data/hypernerf
cd data/hypernerf
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_cut-lemon.zip 

unzip interp_cut-lemon.zip 
mv cut-lemon interp_cut-lemon # make sure the folder name begins with interp_ or vrig_ ...
```


## Training
We provide the training script for TimeFormer, here we use [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) as the backbone.
```shell
# Training with TimeFormer
python train_timeformer.py -s data/interp_cut-lemon1 -m output/interp_cut-lemon1 \
    --eval --iterations 20000 \
    --n_layer 4 --weight_reg 0.0001 --weight_t 0.8 --batch 4

# Rendering
python render.py -m output/interp_cut-lemon1_batch_4_trans_0.8_reg_0.0001_layer_4_random --mode render --eval --skip_train

# Evaluation
python metrics.py -m output/interp_cut-lemon1_batch_4_trans_0.8_reg_0.0001_layer_4_random
```
