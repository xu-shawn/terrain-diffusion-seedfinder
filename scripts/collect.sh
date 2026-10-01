#!/bin/bash
# Pull worker output from every box in hosts.txt ("NAME ip:port" per line) into prod/NAME/.
# Usage: scripts/collect.sh [crops]   — "crops" also pulls the kept elevation crops and scan records.
cd "$(dirname "$0")/.."
while read -r name hp; do
  [ -z "$name" ] && continue
  mkdir -p "prod/$name"
  SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p ${hp##*:}"
  rsync -az -e "$SSH" "root@${hp%%:*}:/root/td/work/prod/results.jsonl" "root@${hp%%:*}:/root/td/work/prod.log" "prod/$name/" < /dev/null
  if [ "${1:-}" = "crops" ]; then
    rsync -az -e "$SSH" "root@${hp%%:*}:/root/td/work/prod/crops" "root@${hp%%:*}:/root/td/work/prod/recs" "prod/$name/" < /dev/null
  fi
  echo "$name: $(wc -l < "prod/$name/results.jsonl") results; $(tail -1 "prod/$name/prod.log" | cut -c1-120)"
done < hosts.txt
