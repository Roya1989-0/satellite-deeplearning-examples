# U-Net Semantic Segmentation on Sentinel-2

This is a small, self-contained example where I train a U-Net model on Sentinel-2 imagery for urban land-cover segmentation.

Goal: pixel-wise classification into 4 classes:
- Built-up
- Vegetation
- Bare soil
- Water / other

Main stack
- Python, PyTorch
- rasterio for reading Sentinel-2 bands
- albumentations for data augmentation
- IoU / F1 for evaluation

The example is intentionally lightweight and focused on clarity:  
load Sentinel-2 tiles → create 256×256 patches → train U-Net → evaluate and visualize predictions.

See:
- train_unet_sentinel2.ipynb – end-to-end notebook
- train_unet.py – minimal training script
