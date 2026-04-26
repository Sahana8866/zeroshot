# Zero-Shot Malware Traffic Recognition Across Encrypted Protocols

**Python 3.8+ | PyTorch 2.0+ | 91.08% F1-Score**

Presented at the 3rd International Conference on Computing Systems and Intelligent Applications (ComSIA-2026)
Access the paper here : https://drive.google.com/file/d/1it22mZRUSC5E8VO1tYvwYwRLXhmVWyil/view?usp=sharing

## Problem Statement

Traditional malware detection methods rely on signature-based or supervised learning approaches. These methods fail when encountering:

- New, unseen botnet families (zero-shot scenario)
- Encrypted traffic where payload inspection is impossible
- Evolving malware that changes signatures rapidly

This work addresses the zero-shot malware detection problem: identifying malicious network flows from botnet families that the model has never seen during training.

**Approach:** Learn behavioral patterns instead of family-specific signatures. Network flows are converted to semantic tokens, and the model learns to recognize malicious behavior regardless of the family.

---

## Key Results

### Overall Zero-Shot Test Performance

| Metric | Score |
|--------|-------|
| F1-Score | 0.9108 |
| Precision | 0.9321 |
| Recall | 0.8904 |
| Accuracy | 0.9188 |

### Per-Family Detection Rate (Unseen Botnets)

| Unseen Family | Detection Rate |
|---------------|----------------|
| Neris2 | 89.2% |
| Rbot2 | 91.4% |
| Virut2 | 87.3% |
| Rbot4 | 93.1% |
| Average | 90.3% |

### Confusion Matrix

| Actual \ Predicted | Normal | Malware |
|--------------------|--------|---------|
| Normal | 5661 | 339 |
| Malware | 573 | 4657 |

False Positive Rate: 5.65% | False Negative Rate: 10.96%

---

## Dataset: CTU-13

The CTU-13 dataset contains 13 real botnet scenarios with mixed normal and malicious traffic.

| Split | Families | Sample Count |
|-------|----------|--------------|
| Training | Normal, Neris, Rbot, Menti, Virut, Murlo, Sogou, Rbot3, NsisAy | 85,951 |
| Validation | From training families | ~15,000 |
| Zero-Shot Test | Normal (held-out), Neris2, Rbot2, Virut2, Rbot4 | 11,230 |

**Zero-Shot Guarantee:** Test families are completely excluded from training. The model has never seen a single flow from Neris2, Rbot2, Virut2, or Rbot4.

---

## Architecture

The system follows a four-stage pipeline:

`Network Flow -> Behavioral Tokenization -> Seq2Vec Embedding -> CNN-GRU Ensemble -> Classification`

### Stage 1: Behavioral Tokenization

Raw network flow features are converted to semantic tokens, creating a "language" of network behavior.

| Feature | Token Examples |
|---------|----------------|
| Duration | VVSHT (<0.05s), VSHT (<0.2s), MED (<10s), VLNG (>60s) |
| Bytes | TNY (<128B), SML (<512B), HGE (<1MB), VHGE (>1MB) |
| Packets | VFW (<2), FW (<5), MOD (<20), VMNY (>100) |
| Direction | OUT (>75% source), IN (>75% destination), BAL (mixed) |
| Protocol | TCP, UDP, ICMP |
| TLS/SSL | TLS_HI (port 443), TLS_LO (SSL in state) |

**Output:** Sequence of 17 tokens per network flow.

### Stage 2: Seq2Vec Embedding (Self-Supervised)

- Architecture: 2-layer Bidirectional LSTM
- Embedding progression: 128 -> 256 -> 384 dimensions
- Output: 384-dimensional dense behavioral vector

### Stage 3: CNN-GRU Ensemble Detector

| Component | Specification |
|-----------|---------------|
| Conv1D Layer 1 | 384 -> 128 channels, kernel size 3 |
| Conv1D Layer 2 | 128 -> 256 channels, kernel size 3, MaxPool |
| Conv1D Layer 3 | 256 -> 256 channels, kernel size 3, MaxPool |
| BiGRU | 2 layers, 320 hidden dimensions, bidirectional |
| Ensemble Heads | 5 independent MLP classifiers |
| Final Output | Average of 5 heads -> binary classification |

Total Parameters: 4,389,898

### Stage 4: Training Techniques

| Technique | Purpose |
|-----------|---------|
| CutMix Augmentation | Mix segments from different flows to improve generalization |
| Focal Loss + OHEM | Focus training on hard-to-classify examples |
| SMOTE in Embedding Space | Generate synthetic malware samples for class balancing |
| Cosine Annealing with Warm Restarts | Escape local minima during optimization |
| Mixed Precision Training | Reduce memory usage by 50%, increase speed by 2x |
| SpatialDropout | Randomly drop entire tokens to prevent overfitting |
| Early Stopping | Stop training when validation F1 stops improving (patience=25) |

---

## Technical Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Learning Rate | 2e-4 (AdamW) |
| Weight Decay | 1e-4 |
| Gradient Accumulation Steps | 2 |
| Max Epochs | 100 |
| Early Stopping Patience | 25 |
| Minimum Epochs Before Stop | 40 |

### Loss Function Configuration

- Focal Loss Alpha: 0.25
- Focal Loss Gamma: 2.5
- OHEM Keep Ratio: 40% hardest samples
- CutMix Alpha: 1.0 (Beta distribution)

### Hardware Requirements

- GPU: CUDA-capable (8GB+ VRAM recommended)
- RAM: 16GB+
- Storage: 10GB for dataset and models
- Training Time: 2-4 hours on Tesla T4 GPU

---
