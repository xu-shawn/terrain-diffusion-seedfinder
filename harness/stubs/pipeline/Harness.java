package com.github.xandergos.terraindiffusionmc.pipeline;

import com.github.xandergos.terraindiffusionmc.infinitetensor.FloatTensor;
import com.github.xandergos.terraindiffusionmc.world.HeightConverter;
import com.github.xandergos.terraindiffusionmc.world.WorldScaleManager;

import java.io.DataOutputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Headless driver around the mod's own pipeline classes. Dumps raw little-endian arrays
 * so results can be compared against (and used as ground truth for) the search code.
 *
 * Commands:
 *   syn    seed x1 y1 x2 y2 out          synthetic map float32 (5,H,W)
 *   noise  seed y0 x0 h w ch tile out    gaussian noise float32 (ch,h,w)
 *   coarse seed ci0 cj0 ci1 cj1 out      coarse slice float32 (7,H,W), weighted sums + weight
 *   elev   seed i1 j1 i2 j2 out          native elevation float32 (H,W) in meters
 *   blocks seed scale z1 x1 z2 x2 out    Minecraft surface Y int16 (H,W) + biome int16 (H,W)
 */
public final class Harness {

    private static void writeFloats(String path, float[] data) throws Exception {
        ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(data);
        try (FileOutputStream out = new FileOutputStream(path)) {
            out.write(buf.array());
        }
    }

    private static void writeShorts(String path, short[] data) throws Exception {
        ByteBuffer buf = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        buf.asShortBuffer().put(data);
        try (FileOutputStream out = new FileOutputStream(path)) {
            out.write(buf.array());
        }
    }

    public static void main(String[] args) throws Exception {
        String cmd = args[0];
        long seed = Long.parseLong(args[1]);
        long t0 = System.currentTimeMillis();
        switch (cmd) {
            case "syn": {
                int x1 = Integer.parseInt(args[2]), y1 = Integer.parseInt(args[3]);
                int x2 = Integer.parseInt(args[4]), y2 = Integer.parseInt(args[5]);
                SyntheticMapFactory factory = new SyntheticMapFactory(seed);
                writeFloats(args[6], WorldPipeline.flatten3D(factory.sample(x1, y1, x2, y2)));
                break;
            }
            case "noise": {
                int y0 = Integer.parseInt(args[2]), x0 = Integer.parseInt(args[3]);
                int h = Integer.parseInt(args[4]), w = Integer.parseInt(args[5]);
                int ch = Integer.parseInt(args[6]), tile = Integer.parseInt(args[7]);
                writeFloats(args[8], WorldPipeline.flatten3D(
                        GaussianNoisePatch.generate(seed, y0, x0, h, w, ch, tile, tile)));
                break;
            }
            case "coarse": {
                int ci0 = Integer.parseInt(args[2]), cj0 = Integer.parseInt(args[3]);
                int ci1 = Integer.parseInt(args[4]), cj1 = Integer.parseInt(args[5]);
                try (WorldPipeline pipeline = new WorldPipeline(seed)) {
                    FloatTensor slice = pipeline.getCoarseSlice(ci0, cj0, ci1, cj1);
                    writeFloats(args[6], slice.data);
                }
                break;
            }
            case "elev": {
                int i1 = Integer.parseInt(args[2]), j1 = Integer.parseInt(args[3]);
                int i2 = Integer.parseInt(args[4]), j2 = Integer.parseInt(args[5]);
                try (WorldPipeline pipeline = new WorldPipeline(seed)) {
                    writeFloats(args[6], pipeline.get(i1, j1, i2, j2, false)[0]);
                }
                break;
            }
            case "blocks": {
                int scale = Integer.parseInt(args[2]);
                int z1 = Integer.parseInt(args[3]), x1 = Integer.parseInt(args[4]);
                int z2 = Integer.parseInt(args[5]), x2 = Integer.parseInt(args[6]);
                int tile = 256;
                if (Math.floorMod(z1, tile) != 0 || Math.floorMod(x1, tile) != 0
                        || Math.floorMod(z2, tile) != 0 || Math.floorMod(x2, tile) != 0) {
                    throw new IllegalArgumentException("block bounds must be multiples of 256");
                }
                WorldScaleManager.setCurrentScale(scale);
                LocalTerrainProvider.init(seed);
                LocalTerrainProvider provider = LocalTerrainProvider.getInstance();
                int H = z2 - z1, W = x2 - x1;
                short[] heights = new short[H * W];
                short[] biomes = new short[H * W];
                for (int tz = z1; tz < z2; tz += tile) {
                    for (int tx = x1; tx < x2; tx += tile) {
                        // Same call the mod's density function makes: fetchHeightmap(Z, X, Z+256, X+256)
                        LocalTerrainProvider.HeightmapData data =
                                provider.fetchHeightmap(tz, tx, tz + tile, tx + tile);
                        for (int r = 0; r < tile; r++) {
                            for (int c = 0; c < tile; c++) {
                                int idx = (tz - z1 + r) * W + (tx - x1 + c);
                                heights[idx] = (short) HeightConverter.convertToMinecraftHeight(
                                        data.heightmap[r][c], scale);
                                biomes[idx] = data.biomeIds[r][c];
                            }
                        }
                    }
                    System.err.printf("row z=%d done (%.1fs)%n", tz, (System.currentTimeMillis() - t0) / 1000.0);
                }
                short[] all = new short[2 * H * W];
                System.arraycopy(heights, 0, all, 0, H * W);
                System.arraycopy(biomes, 0, all, H * W, H * W);
                writeShorts(args[7], all);
                break;
            }
            default:
                throw new IllegalArgumentException("unknown command " + cmd);
        }
        System.err.printf("%s done in %.1fs%n", cmd, (System.currentTimeMillis() - t0) / 1000.0);
        System.exit(0);
    }
}
