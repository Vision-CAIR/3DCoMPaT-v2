# 🐯 3D Models trained on 3DCoMPaT++

We provide here some 3D models trained on the 3DCoMPaT dataset.
This repo includes the code for 3d Shape Classification and Part Segmentation on 3DCoMPaT dataset for both coarse and fine grained versions using prevalent 3D vision algorithms, including [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf), [DGCNN](https://arxiv.org/abs/1801.07829), [PCT](https://arxiv.org/pdf/2012.09688.pdf), [PointStack](https://arxiv.org/abs/2205.09962), and [CurveNet](https://arxiv.org/abs/2105.01288) in pytorch.

You can find the pretrained models and log files in [gdrive](https://drive.google.com/drive/folders/1V4KsOeG-RaaIYYtIv2hYH9mmjBxg3rW_?usp=sharing).

## **Results**

### **Segmentation**

#### **Fine-grained**

| Model                 | Number of points | Accuracy | Shape-aware mIOU | Shape-agnostic mIOU | ckpt                                                                                            |
| --------------------- | ---------------- | -------- | ---------------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| PCT                   | 2048             | 70.49    | 81.31            | 49.09               | [gdrive](https://drive.google.com/file/d/14naxFZiANc2USB7Dx-8mld9TK1XnXJgC/view?usp=share_link) |
| PointNet2 partseg_ssg | 2048             | 71.09    | 80.01            | 50.39               | [gdrive](https://drive.google.com/file/d/1_dPGJU1n4Q4pzm6ZxSw0W5zYKlP5mOcL/view?usp=share_link) |
| Curvenet              | 2048             | 72.49    | 81.37            | 53.09               | [gdrive](https://drive.google.com/file/d/1rmGNvb2uXPLqSDtOU7wGiz9Q_4nozd1D/view?usp=share_link) |
| PointNeXt             | 2048             | 82.07    | 83.92            | 63.72               | [gdrive](https://drive.google.com/file/d/1ABNbcde2gMu0IU6Eub3BXkNr2HHvHFqN/view?usp=share_link) |

#### **Coarse-grained**

| Model                 | Number of points | Accuracy | Shape-aware mIOU | Shape-agnostic mIOU | ckpt                                                                                            |
| --------------------- | ---------------- | -------- | ---------------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| PCT                   | 2048             | 80.64    | 75.48            | 66.95               | [gdrive](https://drive.google.com/file/d/1xvQ_kQv5lGHMddL4ulhUnXQlHuHRAgaQ/view?usp=share_link) |
| PointNet2 partseg_ssg | 2048             | 84.72    | 77.98            | 73.79               | [gdrive](https://drive.google.com/file/d/1HrwGvEr3RUq2KNKZCdPhho1y-krTmwwt/view?usp=share_link) |
| Curvenet              | 2048             | 86.01    | 80.64            | 76.32               | [gdrive](https://drive.google.com/file/d/1Q6yhwFemwIVcy1RivSIXJ0z_8Fz986I0/view?usp=share_link) |
| PointNeXt             | 2048             | 94.17    | 86.80            | 85.45               | [gdrive](https://drive.google.com/file/d/174EHOftBhupCRI3p-vRjQayB4Z_Z1rGG/view?usp=share_link) |

### **Classification**

| Model             | Number of points | Accuracy | ckpt                                                                                            |
| ----------------- | ---------------- | -------- | ----------------------------------------------------------------------------------------------- |
| DGCNN             | 2048             | 78.85    | [gdrive](https://drive.google.com/file/d/185Y7Qr4tTavhYulHAAj2O_1MEBOM19A-/view?usp=share_link) |
| PCT               | 2048             | 68.88    | [gdrive](https://drive.google.com/file/d/1cFiuVSXI5TKjEtkjuvHEBQCl54-FCnsq/view?usp=share_link) |
| PointNet2 cls_msg | 2048             | 84.10    | [gdrive](https://drive.google.com/file/d/1QKSVoM0sIuhMQjnsxe3YyA2TM-HnQr4u/view?usp=share_link) |
| PointStack        | 2048             | 83.04    | [gdrive](https://drive.google.com/file/d/1MYEXcZASDvkDsfliW6aZUGK1yJXFyorM/view?usp=share_link) |
| Curvenet          | 2048             | 85.14    | [gdrive](https://drive.google.com/file/d/1yfTPsi0hCtFAqA94hGfYEtc0vbL-WXz6/view?usp=share_link) |
| PointNeXt         | 2048             | 83.01    | [gdrive](https://drive.google.com/file/d/1E_lotEnXb-lhiBug1iyXw51UE7JD5xZS/view?usp=share_link) |
