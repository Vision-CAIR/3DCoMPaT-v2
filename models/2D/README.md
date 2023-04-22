# 🦁 2D Models trained on 3DCoMPaT++

We provide here some 2D models trained on the 3DCoMPaT dataset.
For all models, you must include the loader files from the `repo_root/loaders/2D` in the active folder (you can simply use a symlink).

## **Results**

### Shape classification

#### Fine Grained

| Model    | Accuracy Top 1 | Accuracy Top 5 | ckpt                                                                               |
| -------- | -------------- | -------------- | ---------------------------------------------------------------------------------- |
| resnet18 | 76.26          | 97.06          | [gdrive](https://drive.google.com/drive/folders/1ItgvBBaySnFSrNYIsMufDtKV_kYqIfB-) |
| resnet50 | 90.20          | 99.09          | [gdrive](https://drive.google.com/drive/folders/1mNcUyLMqaVg1xHZ1IUhUDH3ZO0WNGaPu) |

### **Segmentation**

| Model                 | MIOU  | mPrecision | ckpt                                                                               |
| --------------------- | ----- | ---------- | ---------------------------------------------------------------------------------- |
| SegFormer_Material    | 82.45 | 95.82      | [gdrive](https://drive.google.com/drive/folders/1ItgvBBaySnFSrNYIsMufDtKV_kYqIfB-) |
| SegFormer_Part_Coarse | 73.35 | 99.09      | [gdrive](https://drive.google.com/drive/folders/1mNcUyLMqaVg1xHZ1IUhUDH3ZO0WNGaPu) |
| SegFormer_Part_Fine   | 52.24 | 97.01      | [gdrive](https://drive.google.com/drive/folders/1mNcUyLMqaVg1xHZ1IUhUDH3ZO0WNGaPu) |

### **Classification**

#### **Fine Grained**

| Model    | Accuracy Top 1 | Accuracy Top 5 | ckpt                                                                               |
| -------- | -------------- | -------------- | ---------------------------------------------------------------------------------- |
| resnet18 | 76.269         | 97.06          | [gdrive](https://drive.google.com/drive/folders/1ItgvBBaySnFSrNYIsMufDtKV_kYqIfB-) |
| resnet50 | 90.203         | 99.09          | [gdrive](https://drive.google.com/drive/folders/1mNcUyLMqaVg1xHZ1IUhUDH3ZO0WNGaPu) |

#### **Coarse Grained**

| Model    | Accuracy Top 1 | Accuracy Top 5 | ckpt                                                                               |
| -------- | -------------- | -------------- | ---------------------------------------------------------------------------------- |
| resnet18 | 75.435         | 96.68          | [gdrive](https://drive.google.com/drive/folders/116C4EY3GQl3zQp439gzT6Awb76yVhaAF) |
| resnet50 | 90.03          | 99.08          | [gdrive](https://drive.google.com/drive/folders/1HROyrFX5Zlycq7IJxPsUJkCBBptoVjQt) |
