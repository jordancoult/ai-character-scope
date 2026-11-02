#!/bin/bash
# Vast.ai helper (API key in .secrets/vast_api_key, never printed).
#   vast.sh search            US/CA RTX 5090 offers suitable for our pipelines, cheapest first (incl. download cost)
#   vast.sh search-any        same, any country
#   vast.sh create OFFER_ID   rent it (image runpod/pytorch cu128, 150 GB disk, SSH + port 8080, our public key)
#   vast.sh show              my instances (id, status, ssh host/port)
#   vast.sh ssh-addr ID       ssh://root@HOST:PORT for cloud/infra/.pod
#   vast.sh stop ID / start ID   pause (keeps the disk: weights + caches; small storage fee) / resume
#   vast.sh destroy ID        terminate (disk lost)
set -euo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
V="$R/cloud/.venv/bin/vastai"
KEY=$(cat "$R/.secrets/vast_api_key")
Q='num_gpus=1 gpu_name=RTX_5090 compute_cap>=1200 gpu_ram>=30 cpu_ram>=48 disk_space>=150 cuda_vers>=12.8 inet_down>=500 reliability>0.97 direct_port_count>=2 rentable=true'
case "${1:-}" in
  search|search-any)
              # cheapest first, by hourly price + the one-time cost of downloading our ~50 GB of weights
              # (Vast hosts set their own $/GB download price; there's no cross-host storage, so every new host downloads)
              GEO=$([ "$1" = search ] && echo " geolocation in [US,CA]")
              $V --api-key "$KEY" --raw search offers "$Q inet_down_cost<=0.02$GEO" -o 'dph_total' | python3 -c '
import sys, json
o = json.load(sys.stdin)
for x in o: x["dl50"] = 50 * (x.get("inet_down_cost") or 0)
o.sort(key=lambda x: x["dph_total"] + x["dl50"] / 3)   # amortize the download over a ~3 h session
print(f"{"id":>10} {"$/h":>6} {"dl 50GB":>8} {"store/mo":>9} {"down Mbps":>9} {"RAM":>5}  where")
for x in o[:15]:
    print(f"{x["id"]:>10} {x["dph_total"]:6.3f} {x["dl50"]:8.2f} {x.get("storage_cost",0)*150:9.2f} {x.get("inet_down",0):9.0f} {x.get("cpu_ram",0)/1024:5.0f}  {x.get("geolocation")}")
' ;;
  create)     OUT=$($V --api-key "$KEY" --raw create instance "$2" --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
                --disk 150 --ssh --direct --env "-p 8080:8080")
              echo "$OUT"
              ID=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
              $V --api-key "$KEY" attach ssh "$ID" "$(cat "$R/.secrets/runpod_ed25519.pub")"   # authorize our key on this instance
              echo "instance $ID" ;;
  ssh-addr)   $V --api-key "$KEY" ssh-url "$2" ;;
  show)       $V --api-key "$KEY" show instances ;;
  stop)       $V --api-key "$KEY" stop instance "$2" ;;
  start)      $V --api-key "$KEY" start instance "$2" ;;
  destroy)    $V --api-key "$KEY" destroy instance "$2" ;;
  *) sed -n 2,8p "$0" ;;
esac
