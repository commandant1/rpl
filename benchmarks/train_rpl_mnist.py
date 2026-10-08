#!/usr/bin/env python3
"""Train a small MLP on MNIST using the local `rpl` package.

Writes JSON with timings and accuracy to `--out`.
"""
import argparse, time, json
import numpy as np
import os, urllib.request
import rpl
import rpl.nn as nn
import rpl.optim as optim
import threading, time as _time
import psutil

def one_hot(y, k=10):
    out = np.zeros((y.shape[0], k), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def _download_mnist(path):
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    print('Downloading MNIST...')
    urllib.request.urlretrieve(url, path)

def load_mnist():
    # Try torchvision first
    try:
        from torchvision import datasets, transforms
        dtr = datasets.MNIST('.', train=True, download=True)
        dte = datasets.MNIST('.', train=False, download=True)
        x_train = dtr.data.numpy()
        y_train = dtr.targets.numpy()
        x_test = dte.data.numpy()
        y_test = dte.targets.numpy()
        return (x_train, y_train), (x_test, y_test)
    except Exception:
        cache = os.path.expanduser('~/.cache/mnist.npz')
        if not os.path.exists(cache):
            _download_mnist(cache)
        with np.load(cache, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

def train(epoch_count=5, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    import os
    subset = int(os.environ.get('MNIST_SUBSET', '0'))
    if subset > 0:
        x_train = x_train[:subset]; y_train = y_train[:subset]
        x_test = x_test[:min(len(x_test), max(256, subset//6))]; y_test = y_test[:len(x_test)]
    x_train = x_train.reshape(-1, 28*28).astype(np.float32)/255.0
    x_test = x_test.reshape(-1, 28*28).astype(np.float32)/255.0
    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    # Model: Linear(784->128) -> ReLU -> Linear(128->10)
    l1 = nn.Linear(28*28, 128)
    l2 = nn.Linear(128, 10)
    params = [l1.weight, l1.bias, l2.weight, l2.bias]
    opt = optim.SGD(params, lr=0.01)

    num_batches = int(np.ceil(x_train.shape[0] / batch_size))
    # start memory monitor
    max_rss = {'v': 0}
    stop_mon = {'v': False}
    def _monitor():
        p = psutil.Process()
        while not stop_mon['v']:
            rss = p.memory_info().rss
            if rss > max_rss['v']: max_rss['v'] = rss
            _time.sleep(0.05)
    th = threading.Thread(target=_monitor, daemon=True)
    th.start()

    start_time = time.time()
    for epoch in range(epoch_count):
        idx = np.random.permutation(len(x_train))
        for b in range(num_batches):
            bs = b*batch_size
            be = min(bs+batch_size, len(x_train))
            xb = x_train[idx[bs:be]]
            yb = y_train_oh[idx[bs:be]]

            xt = rpl.Tensor(xb, requires_grad=True)
            out = l2(l1(xt).relu())
            tgt = rpl.Tensor(yb)
            loss = nn.mse_loss(out, tgt)
            loss.backward()
            opt.step()
            opt.zero_grad()

    train_time = time.time() - start_time
    stop_mon['v'] = True
    th.join()

    # Evaluate
    logits = []
    for i in range(0, x_test.shape[0], batch_size):
        xb = x_test[i:i+batch_size]
        xt = rpl.Tensor(xb)
        out = l2(l1(xt).relu())
        logits.append(out.data)
    logits = np.vstack(logits)
    preds = np.argmax(logits, axis=1)
    acc = (preds == y_test).mean()

    return {"framework":"rpl","epochs":epoch_count,"batch_size":batch_size,"train_time_s":train_time,"test_acc":float(acc), "max_rss_bytes": int(max_rss['v'])}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset", type=int, default=0, help="use only first N training samples (for quick runs)")
    p.add_argument("--out", type=str, default="rpl_result.json")
    args = p.parse_args()
    if args.subset and args.subset > 0:
        # slice dataset for quick runs
        (x_train, y_train), (x_test, y_test) = load_mnist()
        x_train = x_train[:args.subset]; y_train = y_train[:args.subset]
        x_test = x_test[:min(len(x_test), max(256, args.subset//6))]; y_test = y_test[:len(x_test)]
        # monkeypatch into train by closure
        def train_with_slice(e,b):
            return train(e,b)
        res = train(args.epochs, args.batch_size)
    else:
        res = train(args.epochs, args.batch_size)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(res)

if __name__ == '__main__':
    main()
