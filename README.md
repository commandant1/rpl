# RPL - RPI Learn

**A high-performance machine learning library for Raspberry Pi 4**

Clean Python API • Optimized C Core • ARM NEON SIMD • Autograd • Quantization

---

## 🎯 **What is RPL?**

RPL (RPI Learn) is a hybrid machine learning framework designed for the Raspberry Pi 4. It combines the ease of use of a PyTorch-like Python API with the performance of a pure C core optimized for ARM architecture.

- **Fast**: Hand-tuned ARM NEON SIMD kernels and OpenMP multi-threading.
- **Python-First**: Familiar API for rapid prototyping.
- **Production-Ready**: Export models to standalone C code for zero-overhead deployment.
- **Complete**: Supports Transformers, CNNs, classical algorithms (SVM, KMeans), and RL.

---

## 🐍 **Python Quick Start**

```python
import rpl
import rpl.nn as nn

# Define a model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Training loop
optimizer = rpl.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🚀 **Installation**

### **Prerequisites**
- Raspberry Pi 4 (or ARMv8 system)
- CMake >= 3.10
- Python >= 3.7
- GCC with OpenMP support

### **Building from Source**
```bash
git clone https://github.com/yourusername/rpl.git
cd rpl
mkdir build && cd build
cmake ..
make -j4
```

---

## 📚 **Documentation**

- **[Python API Reference](docs/PYTHON_API.md)** - Getting started with Python
- **[C API Reference](docs/C_API.md)** - Low-level C documentation
- **[Examples](examples/)** - Working examples in C and Python
- **[Performance Guide](docs/COMPARISON.md)** - Benchmarks vs PyTorch/TF

---

## 🏆 **Performance Benchmarks**

| Operation | Size | RPL | PyTorch | TensorFlow |
|-----------|------|-----|---------|------------|
| **GEMM (FP32)** | 512×512 | **18 GFLOPS** | 12 GFLOPS | 10 GFLOPS |
| **GEMM (INT8)** | 512×512 | **60 GOPS** | N/A | 45 GOPS |
| **Startup Time** | - | **<0.1s** | 3.5s | 5.2s |
| **Library Size** | - | **200 KB** | 1.5 GB | 500 MB |

*Benchmarked on Raspberry Pi 4 (4GB RAM)*

---

## 🎨 **Feature Set**

### **Deep Learning**
- Layers: Linear, Conv2D, Conv3D, LSTM, GRU, Multi-Head Attention, BatchNorm
- Activations: ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU
- Autograd: Fully automatic differentiation via a dynamic computation graph

### **Classical Machine Learning**
- Classification: SVM, Naive Bayes, Logistic Regression
- Clustering: K-Means, DBSCAN
- Dimensionality Reduction: PCA

### **Reinforcement Learning**
- Deep Q-Network (DQN)
- Policy Gradients / Actor-Critic
- Replay Buffers and Environment Wrappers

---

## 🤝 **Contributing**

We welcome contributions! Please see `CONTRIBUTING.md` for our code of conduct and development process.

## 📄 **License**

Published under the MIT License. See `LICENSE` for details.

---

**Made with ❤️ for the embedded ML community**
