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

## Vast.ai specifics (account set up 2026-11-02; helper `cloud/infra/vast.sh`, key in `.secrets/vast_api_key`)
- **No cross-host storage** (no network volumes). Instance disk survives `stop`, is lost on `destroy`; "local volumes"
  are pinned to one physical host. So a new host downloads the ~50 GB of weights again (+ artifacts from HF).
- **Download price is per host** (`inet_down_cost`, $/GB): $0–0.04/GB seen → $0–2 per fresh setup.
  `vast.sh search` filters `inet_down_cost<=0.02` and ranks by $/h + the 50 GB download amortized over 3 h.
- **Stopping is not free:** `storage_cost` × 150 GB = **$30–150/month** ($1–5/day) while stopped. Stop for pauses of
  hours; destroy for pauses longer than ~a day (re-setup ≈ 10–15 min + $0–0.50 download on a cheap host).
- SSH: `--ssh --direct`; our key is attached per instance (`vastai attach ssh`). Port 8080 mapped with `-p 8080:8080`,
  but we use an SSH tunnel anyway.
