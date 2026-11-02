#!/bin/bash
# Save the warm local compile cache as ONE tar under $JOYAI_DATA/cache/<GPU>.tar (a network volume is very slow
# with the ~60k small cache files; one sequential write takes seconds). run_server.sh unpacks it at start;
# publish_artifacts.sh uploads it so any machine with the same GPU model starts warm.
set -e
W=${JOYAI_DATA:-/workspace}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
mkdir -p "$W/cache"
tar -cf "$W/cache/$GPU.tar.tmp" -C "/root/jcache/$GPU" . && mv "$W/cache/$GPU.tar.tmp" "$W/cache/$GPU.tar"
ls -la "$W/cache/$GPU.tar"
