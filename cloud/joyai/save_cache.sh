#!/bin/bash
# Save the warm local compile cache to the volume as ONE tar (the network volume is very slow with the tens of
# thousands of small cache files; a single sequential write takes seconds). run_server.sh unpacks it at start.
set -e
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
mkdir -p /workspace/cache
tar -cf /workspace/cache/$GPU.tar.tmp -C /root/jcache/$GPU . && mv /workspace/cache/$GPU.tar.tmp /workspace/cache/$GPU.tar
ls -la /workspace/cache/$GPU.tar
