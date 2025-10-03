# 🧪 Vision Transformer Experiments on CIFAR-10

This repository documents a series of systematic experiments conducted on CIFAR-10 to understand the training dynamics and performance behavior of Vision Transformers (ViTs) and their hybrid variants. The objective was not to reach state-of-the-art accuracy, but to explore **how different architectural and training modifications influence model performance** under limited compute.

---

## 1️⃣ Baseline ViT

- Implemented a **custom Vision Transformer** from scratch: patch embedding, positional embeddings, class token, and Transformer encoder blocks with pre-layer normalization and learnable gamma scaling.  
- Trained with standard AdamW and cosine learning rate schedule.  
- **Accuracy:** ~80%  
- **Observation:** Training was stable, but accuracy plateaued early. Smaller patch sizes increased compute cost but didn’t yield consistent accuracy improvements.

---

## 2️⃣ Data Augmentation & Regularization

- Added **Mixup** and **CutMix** per batch.  
- Used **RandomCrop**, **RandomHorizontalFlip**, **ColorJitter**, and **RandomErasing**.  
- Introduced **DropPath (stochastic depth)** and label smoothing.  
- **Accuracy:** Slight improvement over baseline (~82–83%).  
- **Observation:** These regularization techniques stabilized training and improved generalization, but the **loss consistently oscillated around 1.2**, indicating a plateau.

---

## 3️⃣ Hybrid CNN + Transformer

- Replaced the patch embedding with a **ResNet-18 backbone** to extract convolutional features.  
- Fed the resulting feature maps into a Transformer encoder (PyTorch implementation).  
- **Accuracy:** ~82%  
- **Observation:** CNN features helped representation, but did not dramatically surpass the ViT baseline. The hybrid architecture slightly improved stability but the loss curve showed similar plateau behavior.

---

## 4️⃣ Explicit Transformer Implementation

- Replaced the built-in Transformer API with **manually coded attention and MLP blocks**, closely following the original ViT paper structure.  
- Incorporated **Pre-Norm**, learnable scaling (γ), and residual connections carefully.  
- **Accuracy:** Similar to hybrid model (~83–84%), but allowed **more controlled experimentation** with block design.

---

## 5️⃣ Knowledge Distillation

- Used a **ResNet-18 teacher** trained on CIFAR-10.  
- ViT as the **student**, trained using KL divergence + cross-entropy loss with temperature scaling.  
- **Accuracy:** ~86%  
- **Observation:** Distillation led to a noticeable boost. The student model benefited from the richer supervision of the teacher, improving generalization without increasing the student’s complexity.

---

## 📊 Summary of Results

| Experiment                          | Accuracy |
|--------------------------------------|----------|
| Baseline ViT                         | ~80%     |
| + Regularization & Augmentations     | 82–83%   |
| Hybrid CNN + Transformer             | ~82%     |
| Explicit Transformer Implementation  | 83–84%   |
| Knowledge Distillation               | **~86%** |

---

## ✨ Key Insights

- **Patch size** affects both compute and accuracy — there’s a sweet spot for small datasets like CIFAR-10.  
- Standard ViT struggles to go beyond ~80% without strong regularization or supervision.  
- Adding a CNN stem provides moderate benefits but doesn’t fundamentally solve optimization plateaus.  
- Knowledge distillation is the most impactful technique among the ones tried, significantly improving performance with minimal student-side changes.

---

