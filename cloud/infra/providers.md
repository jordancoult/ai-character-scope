# GPU providers (how to rent a machine for a pipeline)

Pipelines only need: Linux + NVIDIA driver (CUDA ≥ 12.8), a supported GPU (current pipelines: sm_120 Blackwell,
≥ 32 GB: RTX 5090 or RTX PRO 6000), root, SSH, TCP port 8080 reachable (or an SSH tunnel), ≥ 120 GB disk.

| Provider | RTX 5090 | RTX PRO 6000 | Notes |
|---|---|---|---|
| **RunPod** | $0.69 community / $1.19 secure | $1.69 / $2.49 | MCP tools available in Claude Code (`mcp__runpod__*`). SSH via `PUBLIC_KEY` env. No UDP (fine: we use WebSocket/TCP). Network volumes need Secure Cloud and pin you to one data center. 5090 stock is spotty (2026-11-02: only Europe, on and off) |
| **Vast.ai** | ~$0.40–0.65 (marketplace) | ~$1.15–1.55 | Many US hosts; UDP supported; per-instance disk. Not set up yet (needs an account + API key) |
| TensorDock | ~$0.56–0.65 | — | VMs with open ports; not tried |
| Lambda / Modal | none | Modal $3.03 | Only for B200 (needs an sm_100 build) |

Prices from `experiments/2026-11-01-cloud-gpu-research.md` and RunPod catalog reads on 2026-11-02.

## RunPod specifics
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Python 3.12, uv, sshd; no nvcc, `setup.sh` adds it).
- Create with ports `22/tcp,8080/http`, env `PUBLIC_KEY=<.secrets/runpod_ed25519.pub>`, `gpu.minCudaVersion: "12.8"`.
- `ssh.direct` (host + port) appears in `get-pod` ~30 s after RUNNING.
- Existing network volume `ai-character-cloud` (`jg2ckg7vlg`, 150 GB, EU-RO-1, ~$10.50/month) holds weights, wheels
  and a 5090 cache tar. It's optional now that artifacts live on HF; delete it once the HF path is proven.

## Latency by region
The user is in the US. Measured: laptop → EU-RO-1 adds ~0.2–0.3 s over the 0.46 s pipeline latency. Prefer a US
host when stock allows.
