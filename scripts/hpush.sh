#!/bin/bash
# Usage: scripts/hpush.sh ip:port   — copy the code to /root/td on one GPU box.
hp="$1"; cd "$(dirname "$0")/.."
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p ${hp##*:}" --exclude __pycache__ tdsearch scripts "root@${hp%%:*}:/root/td/"
