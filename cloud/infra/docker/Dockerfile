# DRAFT (not built yet): portable JoyAI-Video-Edit server image for RTX 5090 / sm_120 hosts.
# Weights (~65 GB with both DiT checkpoints) stay on a volume mounted at /workspace; everything else is baked in,
# including the prebuilt kernel wheels and (optionally) a warm torch.compile cache, so a fresh pod starts in minutes.
#   docker build -t joyai-live:5090 -f cloud/infra/docker/Dockerfile .     (context = ai-character-scope/, needs wheels/ + cache/)
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV JOYAI_DIR=/opt/JoyAI-Video-Edit DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
RUN curl -sL -o /tmp/kr.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/kr.deb && apt-get update -qq && apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 \
 && rm -rf /var/lib/apt/lists/*
RUN git clone -q https://github.com/jd-opensource/JoyAI-Video-Edit.git /opt/JoyAI-Video-Edit \
 && uv venv -q /root/venv --python 3.12 \
 && uv pip install -q --python /root/venv/bin/python -r /opt/JoyAI-Video-Edit/deploy/requirements.txt pip setuptools wheel "huggingface_hub[hf_transfer]"
# kernel wheels built by setup_pod.sh (copy from the volume's /workspace/wheels into cloud/infra/docker/wheels first)
COPY cloud/infra/docker/wheels/ /tmp/wheels/
RUN uv pip install -q --python /root/venv/bin/python --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels
# optional warm compile cache for the RTX 5090 (from /workspace/cache/NVIDIA_GeForce_RTX_5090)
COPY cloud/infra/docker/cache/ /root/jcache/
COPY cloud/pipelines/joyai/run_server.sh cloud/pipelines/joyai/save_cache.sh /opt/
EXPOSE 8080
CMD ["bash", "/opt/run_server.sh"]
