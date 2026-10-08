#!/usr/bin/env python3
"""Train a small MLP on MNIST using TensorFlow / Keras.

Writes JSON with timings and accuracy to `--out`.
"""
import argparse, time, json
import numpy as np
import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28*28,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

def train(epochs=5, batch_size=128):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28).astype('float32')/255.0
    x_test = x_test.reshape(-1, 28*28).astype('float32')/255.0

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    start = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    t = time.time() - start

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    return {"framework":"tensorflow","epochs":epochs,"batch_size":batch_size,"train_time_s":t,"test_acc":float(acc)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", type=str, default="tf_result.json")
    args = p.parse_args()
    res = train(args.epochs, args.batch_size)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(res)

if __name__ == '__main__':
    main()
