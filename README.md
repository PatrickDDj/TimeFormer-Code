# TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

## [Project page](https://patrickddj.github.io/TimeFormer/) | [Paper](https://arxiv.org/abs/2411.11941)

![image](https://patrickddj.github.io/TimeFormer/images/banner.png)

We propose **TimeFormer**, a Transformer module that implicitly models the motion pattern via Temporal Attention from a learning perspective (right). TimeFormer is plug-and-play to existing deformable 3D Gaussian reconstruction methods and enhances reconstruction results (left and middle) without affecting their original inference speed.



## Method

![image](https://patrickddj.github.io/TimeFormer/images/timeformer.png)

**The Framework of Deformable 3D Gaussians Reconstruction with TimeFormer.** Existing deformable 3D Gaussians framework usually includes the canonical space and the deformation field (first row), we incorporate TimeFormer to capture cross-time relationships and explore motion patterns implicitly (second row). We share weights of two deformation fields to transfer the learned motion knowledge. This allows us to exclude this Auxiliary Training Module during inference.





> Note: TimeFormer can be easily **plug-and-play to all existing and future deformable 3D Gaussian framework!**
>
> All you need include:
>
> 	1. Copy TimeFormer.py
> 	1. Add TimeFormer branch and its loss
> 	1. Add backpropagation of TimeFormer





## Environment
```shell
git clone https://github.com/PatrickDDj/TimeFormer-Code.git --recursive
cd TimeFormer-Code
pip install torch torchvision
pip install -r requirements.txt
```


## Dataset
```shell
mkdir data
cd data
wget https://github.com/google/hypernerf/releases/download/v0.1/interp_cut-lemon.zip 

unzip interp_cut-lemon.zip 
mv cut-lemon1 interp_cut-lemon1 # make sure the folder name begins with interp_ or vrig_ ...
```


## Training
We provide the training script for TimeFormer, here we use [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) as the backbone.
```shell
# Training with TimeFormer
python train_timeformer.py -s data/interp_cut-lemon1 -m output/interp_cut-lemon1 \
    --eval --iterations 20000 \
    --n_layer 4 --weight_reg 0.0001 --weight_t 0.8 --batch 4

# Rendering
python render.py -m output/interp_cut-lemon1_batch_4_trans_0.8_reg_0.0001_layer_4_random \
		--mode render --eval --skip_train

# Evaluation
python metrics.py -m output/interp_cut-lemon1_batch_4_trans_0.8_reg_0.0001_layer_4_random
```
