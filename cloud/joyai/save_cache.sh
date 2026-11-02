#!/bin/bash
# Copy the warm local compile cache back to the network volume (run after the server is READY).
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
mkdir -p /workspace/cache/$GPU && cp -a /root/jcache/$GPU/. /workspace/cache/$GPU/ && du -sh /workspace/cache/$GPU
