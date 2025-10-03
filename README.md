# Q1 – Vision Transformer Experiments on CIFAR-10 

---

## 1️ Baseline ViT

- Implemented a **custom Vision Transformer** from scratch:
  - Patch embedding
  - Positional embeddings
  - Class token
  - Transformer encoder blocks with pre-layer normalization and learnable gamma scaling
- Trained using **AdamW** and a **cosine learning rate schedule**.
- **Accuracy:** ~80%
- **Observation:** Training was stable but accuracy plateaued early. Smaller patch sizes increased compute cost but didn’t consistently improve accuracy.

---

## 2️ Data Augmentation & Regularization

- Techniques used:
  - **Mixup** and **CutMix**
  - **RandomCrop**, **RandomHorizontalFlip**, **ColorJitter**, **RandomErasing**
  - **DropPath (stochastic depth)** and label smoothing
- **Accuracy:** 82–83%
- **Observation:** These techniques stabilized training and improved generalization. However, **loss oscillated around 1.2**, indicating a plateau.

---

## 3️ Hybrid CNN + Transformer

- Replaced patch embedding with a **Basic CNN** to extract features.
- Fed resulting feature maps into a Transformer encoder (PyTorch).
- **Accuracy:** ~82%
- **Observation:** CNN features improved representation but didn’t dramatically outperform baseline. Slightly better stability.

---

## 4️ Explicit Transformer Implementation

- Replaced built-in Transformer API with **manually coded attention and MLP blocks**, mirroring the original ViT paper.
- Included **Pre-Norm**, learnable γ scaling, and careful residual handling.
- **Accuracy:** 83–84%
- **Benefit:** Enabled controlled experimentation with block designs.

---

##  Knowledge Distillation

- **Teacher:** ResNet-18 trained on CIFAR-10  
- **Student:** ViT  
- Training used **KL divergence + Cross-Entropy loss** with temperature scaling.

| Method                                | Accuracy |
|----------------------------------------|----------|
| Baseline ViT                           | ~80%     |
| + Regularization & Augmentations       | 82–83%   |
| Basic CNN + Transformer               | ~82%     |
| Explicit Transformer Implementation    | 83–84%   |
| Knowledge Distillation (ResNet → ViT)  | 86%      |
| ResNet Backbone + Transformer         | 87%      |

**Key Insight:**  
Direct feature extraction from a CNN backbone (ResNet) into a Transformer encoder **outperformed knowledge distillation** in this setup, achieving **~87% accuracy**.

---

##  Key Takeaways

- **Patch size** strongly affects compute vs. accuracy trade-offs.
- Vanilla ViT struggles to exceed 80% without strong regularization or supervision.
- CNN + ViT hybrids offer moderate gains but don’t solve optimization plateaus alone.
- Knowledge distillation helps but may need longer training for full effect.
- **ResNet + Transformer hybrid achieved the best accuracy** among all experiments.

---

# Q2 – Text-Prompted Object Segmentation with CLIPSeg and SAM 2 

This notebook demonstrates **text-driven object segmentation** by combining **CLIPSeg** for coarse localization with **SAM 2** for precise mask refinement.

---

##  Pipeline Overview

1. **Setup**  
   Installs necessary libraries and dependencies.

2. **Load Image**  
   Allows image loading via file upload or URL.

3. **User Input**  
   Accepts a **text prompt** specifying the object to segment.

4. **CLIPSeg Segmentation**  
   Generates an **initial heatmap** highlighting regions likely to contain the prompted object.

5. **Threshold Heatmap**  
   Converts the heatmap into a **binary mask** using a configurable threshold, producing a coarse segmentation.

6. **SAM 2 Segmentation**  
   Uses the CLIPSeg mask or bounding box as a **prompt** for SAM 2, which refines the segmentation to produce a high-quality mask.

7. **Display Results**  
   Overlays the final SAM 2 mask on the original image for visualization.

---

##  Known Limitations

- **CLIPSeg Accuracy**: Initial heatmap may be coarse; impacts SAM 2 quality.
- **Threshold Sensitivity**: Threshold may need manual tuning to avoid empty masks or background noise.
- **Prompt Dependency**: Poor CLIPSeg mask → poor SAM 2 refinement.
- **Single Object Only**: Current version handles only **one text prompt / object**.


---


