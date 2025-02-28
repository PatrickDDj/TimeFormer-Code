# TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

## [Project page](https://patrickddj.github.io/TimeFormer/) | [Paper](https://arxiv.org/abs/2411.11941)
![image](https://patrickddj.github.io/TimeFormer/images/timeformer.png)

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
