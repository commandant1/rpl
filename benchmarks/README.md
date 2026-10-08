# RPL vs PyTorch vs TensorFlow — MNIST MLP benchmark

This folder contains three training scripts that train a small MLP on MNIST and
record wall-clock training time and test accuracy.

Files:
- `train_rpl_mnist.py` — uses the local `rpl` package
- `train_torch_mnist.py` — PyTorch implementation
- `train_tf_mnist.py` — TensorFlow / Keras implementation
- `compare_results.py` — simple aggregator to print results from JSON outputs

Quick run examples (adjust `--epochs` / `--batch-size` as desired):

```bash
# Create venv and install deps (PyTorch install may vary per platform)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run each benchmark (writes JSON to --out)
python3 train_rpl_mnist.py --epochs 5 --batch-size 128 --out rpl_result.json
python3 train_torch_mnist.py --epochs 5 --batch-size 128 --out torch_result.json
python3 train_tf_mnist.py --epochs 5 --batch-size 128 --out tf_result.json

# Compare
python3 compare_results.py rpl_result.json torch_result.json tf_result.json
```

Notes:
- Installing PyTorch and TensorFlow can be heavy; choose CPU builds appropriate for
  your OS. The `requirements.txt` contains minimal hints but you may need platform
  specific installation commands for `torch`.
