#!/bin/bash
# Build the headless harness: the mod's own pipeline classes + small stand-ins for the Fabric-only ones.
# The mod sources are fetched at a pinned tag, not vendored. Override with TD_MOD_SRC=/path/to/terrain-diffusion-mc.
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"
MOD_REPO=https://github.com/xandergos/terrain-diffusion-mc.git
MOD_TAG=v2.2.0   # 23d3f50; terrain code is unchanged from v2.0.4 through 2.3.0-preview
MAVEN=https://repo1.maven.org/maven2

MOD="${TD_MOD_SRC:-$H/build/mod}"
if [ ! -d "$MOD/src" ]; then
  git clone -q --depth 1 --branch "$MOD_TAG" "$MOD_REPO" "$MOD"
fi
PKG=com/github/xandergos/terraindiffusionmc
FROM="$MOD/src/main/java/$PKG"
SRC="$H/build/src/$PKG"
rm -rf "$H/build/src" "$H/out"
mkdir -p "$SRC/pipeline" "$SRC/infinitetensor" "$SRC/world" "$SRC/config" "$H/lib" "$H/out"

cp "$FROM"/infinitetensor/*.java "$SRC/infinitetensor/"
for f in BiomeClassifier EDMScheduler FastNoiseLite GaussianNoisePatch LaplacianUtils LocalTerrainProvider \
         OnnxModel PipelineModels PortableRng SyntheticMapFactory WorldPipeline WorldPipelineModelConfig; do
  cp "$FROM/pipeline/$f.java" "$SRC/pipeline/"
done
cp "$FROM/world/HeightConverter.java" "$SRC/world/"
# Stand-ins for the classes that need Fabric/Minecraft (config file, model download, per-world scale).
cp "$H"/stubs/config/*.java "$SRC/config/"
cp "$H"/stubs/pipeline/*.java "$SRC/pipeline/"
cp "$H"/stubs/world/*.java "$SRC/world/"

fetch() { [ -s "$H/lib/$(basename "$1")" ] || curl -fsSL -o "$H/lib/$(basename "$1")" "$MAVEN/$1"; }
fetch com/microsoft/onnxruntime/onnxruntime/1.20.0/onnxruntime-1.20.0.jar   # same version the mod ships
fetch com/google/code/gson/gson/2.11.0/gson-2.11.0.jar
fetch org/slf4j/slf4j-api/2.0.16/slf4j-api-2.0.16.jar
fetch org/slf4j/slf4j-simple/2.0.16/slf4j-simple-2.0.16.jar

javac -nowarn -d "$H/out" -cp "$H/lib/*" $(find "$H/build/src" -name '*.java')
echo "harness built: $H/out"
