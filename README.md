# SADC-GAN
This is the official PyTorch implementation of “SADC-GAN: A Saliency-aware and Decomposition-consistent Generative Adversarial Network for Multimodal Image Fusion”

## Recommended Enviroment
 - [ ] PyTorch 1.7.0 
 - [ ] h5py 2.7.0   
 - [ ] numpy 1.19.5
 - [ ] pillow 8.3.1 
 - [ ] torchvision 0.8.0

## Train
The datasets for training can be download from [training_dataset](https://pan.baidu.com/s/1JYapAcJnPDiLyUAWxhValw?pwd=SADC).
### For MSRS dataset
Run: 
```python
python train_fusion_model.py --dataset_file=./train_datasets/MSRS_vis_inf_64.h5 --checkpoint_path=./trained_models/MSRS --epochs=30 --batch_size=96
```
### For TNO dataset
Run: 
```python
python train_fusion_model.py --dataset_file=./train_datasets/TNO_vis_inf_64.h5 --checkpoint_path=./trained_models/TNO --epochs=30 --batch_size=96
```
### For Harvard medical dataset
Run: 
```python
python train_fusion_model.py --dataset_file=./train_datasets/Harvard_mri_pet_64.h5 --checkpoint_path=./trained_models/Harvard --epochs=30 --batch_size=96
```

## Test
### For MSRS dataset
Run: 
```python
python test_fusion_model.py --dataset_path=./test_images/MSRS --hasRGB=Vis --save_path=./fusion_results/MSRS --checkpoint=./checkpoint/MSRS/fusion_model_G_MSRS.pth
```

### For TNO dataset
Run: 
```python
python test_fusion_model.py --dataset_path=./test_images/TNO --hasRGB=No --save_path=./fusion_results/TNO --checkpoint=./checkpoint/TNO/fusion_model_G_TNO.pth
```

### For Harvard medical dataset
Run: 
```python
python test_fusion_model.py --dataset_path=./test_images/Harvard --hasRGB=Inf --save_path=./fusion_results/Harvard --checkpoint=./checkpoint/Harvard/fusion_model_G_Harvard.pth
```

If you have any questions, please create an issue.
