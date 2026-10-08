#!/usr/bin/env python3
"""Compare TFLite inference (using tflite-runtime) with RPL inference.

Downloads a small MNIST TFLite model (tries a few known URLs), runs inference
on `--subset` samples, and reports latency, throughput, and peak RSS.
"""
import os, sys, time, json, urllib.request
import numpy as np
import psutil, threading

from tflite_runtime.interpreter import Interpreter

# Ensure the project's `python/` package dir is on sys.path so we import the
# real `rpl` package, not the local `benchmarks/rpl.py` helper file.
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python'))

# Load the RPL Python modules directly from the project's `python/rpl` folder
# to avoid local file shadowing issues (benchmarks/rpl.py).
import importlib.util, types
core_path = os.path.join(pkg_dir, 'rpl', 'core.py')
nn_path = os.path.join(pkg_dir, 'rpl', 'nn.py')

# Create a package module so relative imports inside rpl modules work.
pkg_mod = types.ModuleType('rpl')
pkg_mod.__path__ = [os.path.join(pkg_dir, 'rpl')]
import sys as _sys
_sys.modules['rpl'] = pkg_mod

# Load rpl.core and rpl.nn as proper submodules
spec_core = importlib.util.spec_from_file_location('rpl.core', core_path)
core = importlib.util.module_from_spec(spec_core)
spec_core.loader.exec_module(core)
_sys.modules['rpl.core'] = core

spec_nn = importlib.util.spec_from_file_location('rpl.nn', nn_path)
rpl_nn = importlib.util.module_from_spec(spec_nn)
spec_nn.loader.exec_module(rpl_nn)
_sys.modules['rpl.nn'] = rpl_nn

# Expose a simple `rpl` namespace used below
rpl = _sys.modules['rpl']
rpl.Tensor = core.Tensor
rpl.nn = rpl_nn
rnn = rpl_nn

MNIST_NPZ = os.path.expanduser('~/.cache/mnist.npz')

TFLITE_CANDIDATES = [
    'https://storage.googleapis.com/download.tensorflow.org/models/tflite/mnist/model.tflite',
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.tflite',
    'https://storage.googleapis.com/download.tensorflow.org/models/mnist/model.tflite',
]

def download_file(url, dst):
    print('Downloading', url)
    urllib.request.urlretrieve(url, dst)

def get_tflite_model(path='mnist.tflite'):
    if os.path.exists(path):
        return path
    for url in TFLITE_CANDIDATES:
        try:
            download_file(url, path)
            return path
        except Exception as e:
            print('Failed to download', url, e)
    raise RuntimeError('No tflite model could be downloaded')

def load_mnist():
    if not os.path.exists(MNIST_NPZ):
        print('Downloading MNIST dataset...')
        urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', MNIST_NPZ)
    with np.load(MNIST_NPZ) as f:
        x_test = f['x_test']
        y_test = f['y_test']
    x_test = x_test.reshape(-1, 28*28).astype(np.float32)/255.0
    return x_test, y_test

def measure_peak(func):
    max_rss = {'v':0}
    stop = {'v':False}
    def monitor():
        p = psutil.Process()
        while not stop['v']:
            rss = p.memory_info().rss
            if rss > max_rss['v']: max_rss['v'] = rss
            time.sleep(0.02)
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    t0 = time.time()
    res = func()
    dur = time.time()-t0
    stop['v'] = True
    th.join()
    return res, dur, int(max_rss['v'])

def run_tflite(model_path, x, batch_size=64):
    interp = Interpreter(model_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    def _work():
        preds = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].astype(np.float32)
            # reshape if model expects images
            try:
                interp.set_tensor(inp_idx, xb)
            except Exception:
                # try reshape to (N,28,28,1)
                interp.set_tensor(inp_idx, xb.reshape(-1,28,28,1))
            interp.invoke()
            out = interp.get_tensor(out_idx)
            preds.append(out)
        return np.vstack(preds)

    preds, dur, rss = measure_peak(_work)
    samples = x.shape[0]
    return {'framework':'tflite','samples':samples,'time_s':dur,'throughput_sps':samples/dur,'max_rss_bytes':rss}

def run_rpl(x, batch_size=64):
    # Build RPL model (Linear(784->128)->ReLU->Linear(128->10)) with random weights
    l1 = rnn.Linear(28*28, 128)
    l2 = rnn.Linear(128, 10)
    # random init
    l1.weight.data[:] = np.random.randn(*l1.weight.data.shape).astype(np.float32)
    l1.bias.data[:] = np.random.randn(*l1.bias.data.shape).astype(np.float32)
    l2.weight.data[:] = np.random.randn(*l2.weight.data.shape).astype(np.float32)
    l2.bias.data[:] = np.random.randn(*l2.bias.data.shape).astype(np.float32)

    def _work():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].astype(np.float32)
            xt = rpl.Tensor(xb)
            out = l2(l1(xt).relu())
            _ = out.data
        return True

    res, dur, rss = measure_peak(_work)
    return {'framework':'rpl','samples':x.shape[0],'time_s':dur,'throughput_sps':x.shape[0]/dur,'max_rss_bytes':rss}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--out', type=str, default='tflite_rpl_compare.json')
    args = p.parse_args()

    x_test, y_test = load_mnist()
    x = x_test[:args.subset]

    try:
        model_path = get_tflite_model('mnist_model.tflite')
        tflite_result = run_tflite(model_path, x, batch_size=args.batch_size)
    except Exception as e:
        tflite_result = {'error':'no_model','message': str(e)}
    rpl_result = run_rpl(x, batch_size=args.batch_size)

    res = {'tflite': tflite_result, 'rpl': rpl_result}
    with open(args.out,'w') as f:
        json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2))

if __name__ == '__main__':
    main()
