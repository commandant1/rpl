#!/usr/bin/env python3
"""Train a small MLP on MNIST using PyTorch.

Writes JSON with timings and accuracy to `--out`.
"""
import argparse, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import threading, time as _time, psutil
from torchvision import datasets, transforms

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(epochs=5, batch_size=128, device='cpu'):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Prefer torchvision datasets if available, otherwise load mnist.npz
    try:
        from torchvision import datasets as _datasets
        import os
        subset = int(os.environ.get('MNIST_SUBSET','0'))
        train_ds = _datasets.MNIST('.', train=True, download=True, transform=transform)
        test_ds = _datasets.MNIST('.', train=False, download=True, transform=transform)
        if subset and subset>0:
            train_ds = torch.utils.data.Subset(train_ds, list(range(min(len(train_ds), subset))))
            test_ds = torch.utils.data.Subset(test_ds, list(range(min(len(test_ds), max(256, subset//6)))))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    except Exception:
        import numpy as np, os, urllib.request
        cache = os.path.expanduser('~/.cache/mnist.npz')
        if not os.path.exists(cache):
            urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', cache)
        with np.load(cache, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        # create torch datasets
        x_train = torch.from_numpy(x_train).float().unsqueeze(1)/255.0
        x_test = torch.from_numpy(x_test).float().unsqueeze(1)/255.0
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    model = MLP().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # memory monitor
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

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    t = time.time()-start
    stop_mon['v'] = True
    th.join()

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds.cpu() == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    return {"framework":"pytorch","epochs":epochs,"batch_size":batch_size,"train_time_s":t,"test_acc":acc, "max_rss_bytes": int(max_rss['v'])}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset", type=int, default=0, help="use only first N training samples (for quick runs)")
    p.add_argument("--device", type=str, default='cpu')
    p.add_argument("--out", type=str, default="torch_result.json")
    args = p.parse_args()
    if args.subset and args.subset>0:
        # quick subset mode: pass subset size via env var for loader
        import os
        os.environ['MNIST_SUBSET'] = str(args.subset)
    res = train(args.epochs, args.batch_size, args.device)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(res)

if __name__ == '__main__':
    main()
