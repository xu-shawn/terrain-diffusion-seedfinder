#!/bin/bash
# Usage: harness/run.sh <command> <args...>   (see Harness.java). TD_DEVICE=cpu|gpu
R="$(cd "$(dirname "$0")/.." && pwd)"
exec java -Xmx10g -Dtd.models="$R/models" -Dtd.device="${TD_DEVICE:-cpu}" \
  -Dorg.slf4j.simpleLogger.defaultLogLevel="${TD_LOG:-info}" \
  -cp "$R/harness/out:$R/harness/lib/*" \
  com.github.xandergos.terraindiffusionmc.pipeline.Harness "$@"
