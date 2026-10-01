package com.github.xandergos.terraindiffusionmc.pipeline;

import java.nio.file.Path;

/** Headless stand-in: models are pre-downloaded into the directory given by -Dtd.models. */
public final class ModelAssetManager {
    private static final Path MODEL_DIRECTORY = Path.of(System.getProperty("td.models", "models"));

    private ModelAssetManager() {
    }

    public static void ensureAssetsReady() {
    }

    public static Path resolveAssetPath(String assetFileName) {
        return MODEL_DIRECTORY.resolve(assetFileName);
    }
}
