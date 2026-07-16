# Models From Scratch

This repository contains small, self-contained implementations of classic deep learning models built from first principles. The goal is to make the core math and control flow easy to inspect rather than to provide production-ready training pipelines.

## Repository Structure

```text
model_from_scratch/
|-- ANN/
|   `-- ANN_from_scratch.py
|-- CLIP/
|   `-- Clip_from_scratch.py
|-- CNN/
|   `-- CNN_from_scratch.py
|-- RNN/
|   `-- RNN_from_scratch.py
|-- Transformer/
|   `-- transformer_from_scratch.py
|-- VIT/
|   `-- vit_from_scratch.py
`-- Yolo/
    `-- yolov3_from_scratch.py
```

## Projects

### ANN
Implements a basic artificial neural network with manual forward propagation, activation functions, and parameter updates.

### CNN
Builds a convolutional network for image classification using PyTorch modules and CIFAR-10 data loading.

### Transformer
Contains a from-scratch Transformer implementation with multi-head attention, masking, and encoder/decoder style building blocks.

### ViT
Implements a Vision Transformer that turns image patches into tokens and applies transformer layers over them.

### YOLOv3
Includes YOLOv3-style object detection utilities such as IoU, non-maximum suppression, bounding-box conversion, and dataset helpers.


## Stack
 
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
 
---
 
## Related
 
- 📝 Technical blogs on each architecture → [asthamaurya05.github.io/blogs](https://asthamaurya05.github.io/blogs)
- 📊 Full ML → DL project roadmap → [github.com/AsthaMaurya05](https://github.com/AsthaMaurya05)
---
