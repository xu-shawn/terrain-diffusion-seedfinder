#!/bin/bash
# Usage: scripts/h.sh ip:port <remote command>   — ssh to one GPU box.
hp="$1"; shift
exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p "${hp##*:}" "root@${hp%%:*}" "$@"
