package com.github.xandergos.terraindiffusionmc.config;

/** Headless stand-in for the mod's config class (the original reads a Fabric config file). */
public final class TerrainDiffusionConfig {
    private TerrainDiffusionConfig() {
    }

    public static String inferenceDevice() {
        return System.getProperty("td.device", "cpu");
    }

    public static boolean offloadModels() {
        return Boolean.parseBoolean(System.getProperty("td.offload", "false"));
    }

    public static boolean validateModel() {
        return false;
    }

    public static int spawnSearchInitialSize() {
        return 16;
    }

    public static int spawnSearchMaxSize() {
        return 128;
    }

    public static int tileSize() {
        return 256;
    }
}
