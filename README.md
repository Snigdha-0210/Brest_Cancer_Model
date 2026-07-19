<div align="center">
  <h1>🧬 Breast Cancer Segmentation with NSCGCN 🧬</h1>
  <p>
    <strong>Advanced Deep Learning Pipeline for Breast Cancer Ultrasound Image Segmentation</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version" />
    <img src="https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c?logo=pytorch" alt="PyTorch" />
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  </p>
</div>

<hr/>

## 📖 About This Project

This project focuses on **Breast Cancer Segmentation** from ultrasound images using advanced deep learning techniques. The goal is to accurately delineate breast lesions, which is a critical step for computer-aided diagnosis (CAD) systems and can significantly aid radiologists in their clinical workflows.

### ❓ The Problem
Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are paramount for successful treatment and improved survival rates. Ultrasound imaging is a widely used modality for breast examination due to its non-invasive nature, cost-effectiveness, and real-time capabilities. 

However, reading ultrasound images is highly dependent on the operator's experience, and the images often suffer from speckle noise, low contrast, and ambiguous boundaries. Automated, accurate segmentation of lesions in ultrasound images remains a challenging yet crucial task.

## 🚀 Our Approach: U-Net + NSCGCN

While standard Convolutional Neural Networks (CNNs), like the well-known U-Net, have shown great success in medical image segmentation, they are fundamentally limited by their local receptive fields. They often struggle to capture long-range spatial dependencies and global context, which can be critical for accurately distinguishing lesion boundaries from normal tissue in complex ultrasound images.

To overcome this limitation, this project employs a hybrid architecture: **U-Net integrated with a Non-Local Spatial Context Graph Convolutional Network (NSCGCN)**.

### 🧠 How it Works

1. **Feature Extraction**: The standard U-Net encoder extracts hierarchical local feature maps from the input ultrasound images.
2. **Global Reasoning**: The `NSCGCNModule` (located in `models/nscgcn.py`) is strategically placed at the bottleneck of the U-Net (`models/unet_gcn.py`). Instead of just looking at local patches, the NSCGCN constructs a graph from the feature maps, mapping spatial features into an interaction space. This allows the network to perform global reasoning, capturing long-range dependencies and contextual relationships across the entire image.
3. **Refined Segmentation**: The decoder part of the U-Net then reconstructs the segmentation mask, utilizing both the local features from skip connections and the globally-aware features from the NSCGCN module. This combination results in more accurate and robust segmentation boundaries compared to using purely local convolutional filters.

## 📂 Project Structure

```text
├── dataset.py        # BreastUltrasoundDataset logic (augmentations via Albumentations)
├── train.py          # Main training script (Data loading, Optimization, Scheduler)
├── evaluate.py       # Script to evaluate trained models on the test set
├── run_ablations.py  # Script for running ablation studies
├── models/
│   ├── unet.py       # Standard baseline U-Net architecture
│   ├── gcn.py        # Basic Graph Convolutional Network layer
│   ├── nscgcn.py     # Non-Local Spatial Context Graph Convolutional Network module
│   └── unet_gcn.py   # Hybrid architecture (UNet + NSCGCNModule)
└── utils/
    ├── losses.py     # BCEDiceLoss (BCE + Dice Loss)
    └── metrics.py    # Evaluation metrics (Dice, IoU, Precision, Recall)
```

## 🛠️ Setup and Training

### 1. Data Preparation
Ensure your data splits are placed correctly in the project root:
- `train/`
- `valid/`
- `test/`

### 2. Training the Model
To train the model from scratch, you can run the following command:
```bash
python train.py --model unet_gcn --epochs 50 --batch_size 8 --lr 1e-3
```
> [!NOTE]  
> Make sure to always specify `--model unet_gcn` to ensure the NSCGCN module is utilized during training.

### 3. Checkpoints
During training, the best model weights (based on the validation Dice score) are saved into the `checkpoints/` directory. You can load these weights later for inference or fine-tuning.

## 🔮 Future Work
- Explore different parameters for the `NSCGCNModule` (e.g., number of nodes).
- Try different augmentation techniques in `dataset.py` to make the model more robust.
- Perform a detailed ablation study comparing the baseline `unet` with `unet_gcn`.

## 🤝 Impact
By improving the accuracy of automated breast lesion segmentation, this project aims to contribute to more reliable and efficient computer-aided diagnosis tools, ultimately assisting healthcare professionals in making better-informed, timely decisions for patient care.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
