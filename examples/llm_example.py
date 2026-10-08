#!/usr/bin/env python3
"""
Example: Load and test the 0.8B LLM model.

This script demonstrates:
1. Creating a 0.8B LLM
2. Forward pass on sample token sequences
3. Measuring inference time and peak memory
"""
import sys
import os
import time
import numpy as np
import psutil

# Add parent dirs to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Avoid local module shadowing
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, pkg_dir)

import rpl
from rpl.llm import build_llm


def measure_memory_peak(func):
    """Measure peak memory during function execution."""
    max_rss = {'v': 0}
    stop = {'v': False}
    
    import threading
    def monitor():
        p = psutil.Process()
        while not stop['v']:
            rss = p.memory_info().rss
            if rss > max_rss['v']:
                max_rss['v'] = rss
            time.sleep(0.05)
    
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    
    t0 = time.time()
    result = func()
    duration = time.time() - t0
    
    stop['v'] = True
    th.join()
    
    return result, duration, int(max_rss['v'])


def test_llm():
    """Test LLM model instantiation and forward pass."""
    print("\n" + "="*70)
    print("LLM Model Test")
    print("="*70)
    
    # Model config
    full_model = "--full" in sys.argv
    if full_model:
        vocab_size = 248320
        hidden_dim = 1024
        num_layers = 24
        intermediate_dim = 3584
        max_seq_len = 262144
    else:
        # Default test configuration scaled for Raspberry Pi RAM (<50MB)
        vocab_size = 8192
        hidden_dim = 256
        num_layers = 4
        intermediate_dim = 768
        max_seq_len = 2048
    
    print(f"\nModel Configuration ({'0.8B Full' if full_model else 'Edge/Test Profile'}):")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Intermediate (FFN) dimension: {intermediate_dim}")
    print(f"  Max context length: {max_seq_len:,}")
    
    # Build model
    print("\nBuilding model...")
    def build():
        return build_llm(vocab_size=vocab_size, hidden_dim=hidden_dim, 
                        num_layers=num_layers, intermediate_dim=intermediate_dim,
                        max_seq_len=max_seq_len)
    
    model, build_time, build_mem = measure_memory_peak(build)
    print(f"  Build time: {build_time:.3f}s")
    print(f"  Peak memory: {build_mem / 1e9:.2f} GB")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 64
    
    # Create random token IDs (0 to vocab_size-1)
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    print(f"  Input shape: {token_ids.shape}")
    print(f"  Sample token IDs (first 10): {token_ids[0, :10]}")
    
    def forward():
        # Convert to tensor
        token_tensor = rpl.Tensor(token_ids.astype(np.float32))
        logits = model(token_tensor)
        return logits
    
    try:
        logits, inf_time, inf_mem = measure_memory_peak(forward)
        print(f"  Inference time: {inf_time:.3f}s")
        print(f"  Peak memory: {inf_mem / 1e9:.2f} GB")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output sample (first 5 logits): {logits.data.flatten()[:5]}")
        print("\n✓ Forward pass successful!")
    except Exception as e:
        print(f"\n✗ Forward pass failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Model instantiation: OK")
    print(f"Forward pass: OK" if 'logits' in locals() else "Forward pass: FAILED")
    
    if 'logits' in locals():
        total_time = build_time + inf_time
        print(f"Total elapsed time: {total_time:.3f}s")
        print(f"Peak memory (build): {build_mem / 1e9:.2f} GB")
        print(f"Peak memory (inference): {inf_mem / 1e9:.2f} GB")


if __name__ == '__main__':
    test_llm()
