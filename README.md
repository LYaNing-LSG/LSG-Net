# LSG-Net
# Light weight Structure-Guided Network with Hydra Interaction Attention and Global-Local Gating Mechanism for High-Resolution Image Inpainting

![M}A7C%D %C 5U_3(RHRJ8 7](https://github.com/user-attachments/assets/f17de061-0717-463a-b01d-ac9551e55814)
Overview of the LSG-Net:The structure extraction module extracts lines and edges from the input low-resolution image.The HRSR model is able to reconstruct high-resolution structures.The SETR model combines structural information to restore textures and generate the final high-resolution image inpainting result.
## Preparation
1. Preparing the environment:
* [Python 3.8]
* [Pytorch >=1.4]
* [OpenCV]
* [numpy==1.19.5]
* [scikit-image==0.17.2]
* Other package like pillow, pyyaml,yacs,tqdm,pandas,lpips
  
Once the packages are installed,  clone this repo as follow: 

    git clone [https://github.com/xavysp/DexiNed.git](https://github.com/LYaNing-LSG/LSG-Net.git.)
2. The training masks we used are contained in coco_mask_list.txt and irregular_mask_list.txt,you can download them from [https://github.com/ewrfcas/MST_inpainting](https://github.com/ewrfcas/MST_inpainting).
3. Download the pretrained masked wireframe detection model to the './ckpt' fold: [LSM-HAWP (MST ICCV2021 retrained from HAWP CVPR2020).](https://drive.google.com/drive/folders/1yg4Nc20D34sON0Ni_IOezjJCFHXKGWUW?usp=sharing)
4. Prepare the wireframes:
   
```
python lsm_hawp_inference.py --ckpt_path <best_lsm_hawp.pth> --input_path <input image path> --output_path <output image path> --gpu_ids '0'
```
5. Prepare the Edge dataset:

   Download the Dexnied code [https://github.com/xavysp/DexiNed] to obtain edge-detected images. Then run python marginalizate.py for edge processing to get the refined edge images.
   ```
    python marginalizate.py
   ```
6. Indoor and Places2 Dataset
   We can obtain the relevant dataset from the [ZITS](https://github.com/DQiaole/ZITS_inpainting?tab=readme-ov-file) and split it into training and validation sets.
7.Save the obtained training images, validation images, edge images, irregular mask set, and Coco masks into .txt files. Place these files in the data_flist directory and modify the corresponding paths in config_LSG_places2.yml.
## Eval

Download pretrained models on Places2 [BaiduDrive](https://pan.baidu.com/s/17REkB_IRXa-yi9iSeQ7nMA?pwd=7bk6 
), password:7bk6
## Training
Once the data is prepared, we can proceed with training.
 ```
python TSR_train.py --name places2_continous_edgeline --data_path /root/train/train_list.txt --edge_path /root/train/edge_list.txt --train_line_path /root/train_line --irregular_mask_path /root/ir/irregular.txt --coco_mask_path /root/coco/coco.txt --train_epoch 24 --validation_path /root/train/val_list.txt --val_line_path /root/val_line --valid_mask_path /root/dataroot/test_mask --val_edge_path /root/train/edgeval_list.txt --nodes 1 --gpus 1 --GPU_ids '0' --AMP
```
Training No-edge model First,You can refer to [ZITS](https://github.com/DQiaole/ZITS_inpainting?tab=readme-ov-file) as a reference.

**We reference the code, but please note that our structure is different from LAMA.**
```
python FTR_train.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/places2 --config_file ./config_list/config.yml --lama
```
The model with Edge-assisted training.
```
python FTR_train.py --nodes 1 --gpus 2 --GPU_ids '0,1' --path ./ckpt/places2_HR \
--config_file ./config_list/config_LSG_places2.yml --DDP
```
## Acknowledgments
This repo is built upon [MST](https://github.com/ewrfcas/MST_inpainting)., [ZITS](https://github.com/DQiaole/ZITS_inpainting?tab=readme-ov-file)and [LaMa](https://github.com/advimman/lama).




