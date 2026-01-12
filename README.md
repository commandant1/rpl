# RPL - RPI Learn

**A high-performance machine learning library for Raspberry Pi 4**

Clean Python API • Optimized C Core • ARM NEON SIMD • Autograd • Quantization

---

## 🎯 **What is RPL?**

RPL (RPI Learn) is a hybrid machine learning framework designed for the Raspberry Pi 4. It combines the ease of use of a PyTorch-like Python API with the performance of a pure C core optimized for ARM architecture.

- **Optimized for RPi 4**: Hand-tuned for Cortex-A72 (NEON) and VideoCore VI GPU (GLES Compute).
- **GPU Acceleration**: Matrix multiplication and activations run on the VideoCore VI GPU. [See Hardware Details](docs/HARDWARE.md).
- **Lightweight**: Minimum dependencies (OpenBLAS, optional).
ti-threading.
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

### **Cross Compilation**

To compile for Raspberry Pi (AArch64) from an x86 host:
```bash
# Install cross-compiler (Debian/Ubuntu)
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Build using toolchain
mkdir build-cross && cd build-cross
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-aarch64.cmake -DUSE_GPU=OFF
make -j4
```
Note: GPU support (`USE_GPU=ON`) requires cross-compiled EGL/GLES libraries in your sysroot.

### **Quick Installation**
```bash
git clone https://github.com/commandant1/rpl.git
cd rpl
chmod +x install.sh
./install.sh
```

Alternatively, install via pip:
```bash
pip install .
```

### **Manual Building from Source**
```bash
git clone https://github.com/commandant1/rpl.git
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
