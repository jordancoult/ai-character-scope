#!/bin/bash
# Vast.ai helper (API key in .secrets/vast_api_key, never printed).
#   vast.sh search            US/CA RTX 5090 offers suitable for our pipelines, cheapest first
#   vast.sh search-any        same, any country
#   vast.sh create OFFER_ID   rent it (image runpod/pytorch cu128, 150 GB disk, SSH + port 8080, our public key)
#   vast.sh show              my instances (id, status, ssh host/port)
#   vast.sh ssh-addr ID       ssh://root@HOST:PORT for cloud/infra/.pod
#   vast.sh destroy ID        terminate
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
V="$R/cloud/.venv/bin/vastai"
KEY=$(cat "$R/.secrets/vast_api_key")
Q='num_gpus=1 gpu_name=RTX_5090 compute_cap>=1200 gpu_ram>=30 cpu_ram>=48 disk_space>=150 cuda_vers>=12.8 inet_down>=500 reliability>0.97 direct_port_count>=2 rentable=true'
case "${1:-}" in
  search)     $V --api-key "$KEY" search offers "$Q geolocation in [US,CA]" -o 'dph_total' | head -15 ;;
  search-any) $V --api-key "$KEY" search offers "$Q" -o 'dph_total' | head -15 ;;
  create)     OUT=$($V --api-key "$KEY" --raw create instance "$2" --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
                --disk 150 --ssh --direct --env "-p 8080:8080")
              echo "$OUT"
              ID=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
              $V --api-key "$KEY" attach ssh "$ID" "$(cat "$R/.secrets/runpod_ed25519.pub")"   # authorize our key on this instance
              echo "instance $ID" ;;
  ssh-addr)   $V --api-key "$KEY" ssh-url "$2" ;;
  show)       $V --api-key "$KEY" show instances ;;
  destroy)    $V --api-key "$KEY" destroy instance "$2" ;;
  *) sed -n 2,8p "$0" ;;
esac
