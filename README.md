# Semantic Similarity Between Legal Clauses — Model Comparison Report

## Objective
To develop two NLP models capable of identifying **semantic similarity between legal clauses**, trained **from scratch** without using any pre-trained transformer or fine-tuned legal model.

Two baseline models were implemented:

1. **BiLSTM Siamese Network**
2. **Siamese CNN**

---

## Dataset and Splits
- **Dataset:** Legal clause pairs labeled as *similar* or *dissimilar*  
- **Total Clause Types:** 350+ unique clause categories  
- **Data Split:**
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%  
- All clauses were tokenized, encoded, and padded to a uniform sequence length.

---

## Network Details

###  1. BiLSTM Siamese Network
- **Embedding:** 300-dimensional trainable embeddings  
- **BiLSTM Layers:** 2 layers with 128 hidden units  
- **Dropout:** 0.3  
- **Fully Connected Layer:** 64 → 1 (sigmoid activation)  
- **Similarity Metric:** Manhattan distance  
- **Optimizer:** Adam (lr = 1e-3)  
- **Loss:** Binary Cross Entropy  
- **Epochs:** 20  
- **Batch Size:** 64  

###  2. Siamese CNN
- **Embedding:** 300-dimensional trainable embeddings  
- **Convolution Layers:**
  - Conv1D: 128 filters, kernel size = 3  
  - Conv1D: 64 filters, kernel size = 5  
- **Pooling:** Global Max Pooling  
- **Dense Layers:** 64 → 32 → 1 (sigmoid activation)  
- **Similarity Metric:** Cosine distance  
- **Optimizer:** Adam (lr = 1e-3)  
- **Loss:** Binary Cross Entropy  
- **Epochs:** 20  
- **Batch Size:** 64  

---

## Training Setup
- **Early Stopping** enabled based on validation loss.  
- **GPU-accelerated** training for efficiency.  
- Best-performing weights saved automatically during training.  

---

## Performance Metrics

| Metric | BiLSTM | Siamese CNN |
|:--|:--:|:--:|
| **Accuracy** | 0.8710 | **0.9497** |
| **Precision** | 0.8274 | **0.9238** |
| **Recall** | 0.9360 | **0.9798** |
| **F1-Score** | 0.8784 | **0.9510** |
| **ROC-AUC** | 0.9267 | **0.9824** |
| **PR-AUC** | 0.9143 | **0.9790** |

---

## Training Curves

**Figure 1:** *BiLSTM Training and Validation Loss over Epochs*  
> The BiLSTM model shows gradual convergence but mild oscillations in validation loss after epoch 12 — suggesting slight overfitting.
> <img width="1488" height="490" alt="image" src="https://github.com/user-attachments/assets/ec40251c-17c9-4882-87c9-959f9c4ee219" />


**Figure 2:** *Siamese CNN Training and Validation Loss over Epochs*  
> The CNN model converges faster and more smoothly, with stable validation loss and excellent generalization.
<img width="1488" height="490" alt="image" src="https://github.com/user-attachments/assets/f5bc8ead-17dd-4b41-9702-39c336ce68f6" />


---

## Performance Comparison

| Aspect | BiLSTM | Siamese CNN |
|:--|:--|:--|
| **Accuracy** | Moderate | **High** |
| **Training Time** | Slower (sequential) | **Faster** (parallelizable) |
| **Contextual Capture** | Strong long-range dependencies | Focused on local semantic patterns |
| **Overfitting Tendency** | Mild | **Minimal** |
| **Inference Speed** | Slower | **Faster** |
| **Generalization** | Good | **Excellent** |

---

## Analysis and Discussion

### **BiLSTM Strengths**
- Captures **long-term dependencies** and deeper semantic meaning.  
- High **recall** — identifies most semantically similar clauses.  
- Suitable for **multi-sentence legal logic** or condition-heavy clauses.

**Weaknesses:**
- Slower training due to sequential computations.  
- Slight overfitting observed in later epochs.  
- Less efficient for short or locally contextual clauses.

---

### **Siamese CNN Strengths**
- **Superior performance** in all major metrics.  
- **Efficient training/inference** — parallelizable and GPU-friendly.  
- Learns **phrase-level semantics** effectively.  
- Stable training curves and strong generalization.

**Weaknesses:**
- Limited ability to model **long-range dependencies**.  
- Kernel size tuning required for optimal performance.

---

## Conclusion

Both architectures achieve strong results for legal clause similarity detection.  
However, **Siamese CNN** outperforms **BiLSTM** in terms of accuracy, robustness, and computational efficiency.

- **BiLSTM:** Best for complex sequential semantics.  
- **Siamese CNN:** Best overall model for deployment.

### **Final Recommendation**
Deploy **Siamese CNN** as the production model — it achieved:
- **F1 = 0.9510**
- **ROC-AUC = 0.9824**
- **Stable convergence**
- **High inference efficiency**

---

## Suggested Figures for README

| Figure # | Title | Description |
|:--|:--|:--|
| **Figure 1** | *BiLSTM Training Curve* | Training vs Validation Loss (Epoch-wise) |
| **Figure 2** | *Siamese CNN Training Curve* | Training vs Validation Loss (Epoch-wise) |
| **Figure 3** | *ROC Curve Comparison* | ROC curves for both models |
| **Figure 4** | *Precision-Recall Curves* | PR curve comparison |

---
