# Vision Transformer (ViT) for CIFAR-10

A simple implementation of a **Vision Transformer (ViT)** using **PyTorch** for image classification on the **CIFAR-10** dataset.

## Features

- Vision Transformer architecture
- Patch Embedding using Conv2D
- Multi-Head Self Attention
- Transformer Encoder Blocks
- Classification Token (CLS Token)
- Positional Embeddings
- CIFAR-10 dataset support
- Adam optimizer
- CrossEntropyLoss

---

## Project Structure

```
.
├── Vision_Transformer.py
├── data/
└── README.md
```

---

## Requirements

Install the required libraries:

```bash
pip install torch torchvision
```

---

## Dataset

The model is trained on the CIFAR-10 dataset.

- 60,000 RGB images
- Image size: 32 × 32
- 10 classes

The dataset is automatically downloaded by torchvision.

---

## Model Architecture

```
Input Image (3×32×32)
        │
        ▼
Patch Embedding (Conv2D)
        │
        ▼
Flatten Patches
        │
        ▼
Add CLS Token
        │
        ▼
Add Positional Embedding
        │
        ▼
6 Transformer Encoder Blocks
        │
        ▼
CLS Token Output
        │
        ▼
Linear Classifier
        │
        ▼
10 Classes
```

---

## Patch Embedding

The image is divided into patches using:

- Kernel Size = 8
- Stride = 8

For a 32×32 image:

- Number of patches = (32 / 8)² = 16
- Embedding Dimension = 128

---

## Transformer Encoder

Each encoder block contains:

- Multi-Head Self Attention
- Residual Connection
- Layer Normalization
- Feed Forward Network
- GELU Activation

---

## Training

Optimizer:

```python
Adam
```

Loss Function:

```python
CrossEntropyLoss
```

Epochs:

```
5
```

Batch Size:

```
64
```

---

## Testing

The model computes classification accuracy on the test dataset after training.

---

## Run

```bash
python Vision_Transformer.py
```

---

## Expected Output

```
loss is ...
loss is ...
loss is ...

correctness is XX.XX
```

---

## Future Improvements

- Use learnable positional embeddings with correct dimensions
- Add Dropout
- Add MLP Head
- Data Augmentation
- Learning Rate Scheduler
- Save and Load model weights
- GPU support
- Mixed Precision Training
- TensorBoard logging

---

## Known Issues in Current Code

The current implementation contains a few mistakes that should be fixed:

1. `datasets.CIFAR` should be `datasets.CIFAR10`.

2. Positional embedding should account for the CLS token:

```python
self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, 128))
```

3. The forward method uses:

```python
self.encode(x)
```

It should be:

```python
self.encoder(x)
```

4. Optimizer should use:

```python
model.parameters()
```

instead of

```python
model.parameter()
```

5. The normalization values can be replaced with the standard CIFAR-10 statistics for better performance.

---

## Author

Built using **PyTorch** as a learning implementation of the Vision Transformer (ViT).
