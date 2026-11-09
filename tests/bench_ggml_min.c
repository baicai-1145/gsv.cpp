// Is ggml itself slow on this box, or is the engine misusing it?
// Does one plain f32 mul_mat (the AR's GEMV shape) and one conv via ggml_conv_1d, times
// ggml_backend_graph_compute, and prints GMAC/s. No engine, no fixture, no model file.
//
//   cc -O2 -I<llama.cpp>/ggml/include bench_ggml_min.c -o bench_ggml_min \
//      -L<llama.cpp>/build-cpu-rel/bin -lggml -lggml-cpu -lggml-base -Wl,-rpath,<...>/bin
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
    printf("ggml: %s\n", ggml_version ? ggml_version() : "?");
    printf("simd: avx=%d avx2=%d fma=%d avx512=%d\n",
           ggml_cpu_has_avx(), ggml_cpu_has_avx2(), ggml_cpu_has_fma(), ggml_cpu_has_avx512());

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 1);

    // ---- (a) GEMV: mul_mat [2048 x 512] x [512]  (the AR's ffn2 shape) ----
    {
        struct ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), NULL, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 2048);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_set_input(w); ggml_set_input(x); ggml_set_output(y);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, g);
        ggml_backend_graph_compute(be, g);   // warm
        const int reps = 200;
        double t0 = now_s();
        for (int i = 0; i < reps; i++) ggml_backend_graph_compute(be, g);
        double dt = now_s() - t0;
        const double macs = (double) 512 * 2048 * reps;
        printf("mul_mat f32 512x2048 (GEMV): %.3f ms/call  %.2f GMAC/s\n",
               dt / reps * 1e3, macs / dt / 1e9);
        ggml_gallocr_free(ga); ggml_backend_buffer_free(b); ggml_free(ctx);
    }

    // ---- (b) conv: [IC=128, K=11] over T=4120, OC=128 (dec stage1 shape) ----
    {
        const int IC = 128, K = 11, OC = 128, T = 4120;
        struct ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), NULL, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
        struct ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);
        struct ggml_tensor * y  = ggml_conv_1d(ctx, wt, xt, 1, 0, 1);
        ggml_set_input(wt); ggml_set_input(xt); ggml_set_output(y);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, g);
        ggml_backend_graph_compute(be, g);
        const int reps = 20;
        double t0 = now_s();
        for (int i = 0; i < reps; i++) ggml_backend_graph_compute(be, g);
        double dt = now_s() - t0;
        const double macs = (double) IC * K * OC * T * reps;
        printf("conv_1d 128/11/128/T=4120: %.2f ms/call  %.2f GMAC/s\n",
               dt / reps * 1e3, macs / dt / 1e9);
        ggml_gallocr_free(ga); ggml_backend_buffer_free(b); ggml_free(ctx);
    }

    // ---- (c) q8_0 GEMV: what the AR actually runs (weights q8_0, activations converted) ----
    {
        const int K = 512, N = 2048;
        struct ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), NULL, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_set_input(w); ggml_set_input(x); ggml_set_output(y);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
        // fill the weights with something, quantised to q8_0
        {
            float * src = (float *) malloc(sizeof(float) * K * N);
            for (int i = 0; i < K * N; i++) src[i] = (float) ((i * 37 % 97) - 48) / 48.0f;
            ggml_quantize_chunk(GGML_TYPE_Q8_0, src, (void *) w->data, 0, N, K, NULL);
            free(src);
        }
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, g);
        ggml_backend_graph_compute(be, g);
        const int reps = 200;
        double t0 = now_s();
        for (int i = 0; i < reps; i++) ggml_backend_graph_compute(be, g);
        double dt = now_s() - t0;
        const double macs = (double) K * N * reps;
        printf("mul_mat q8_0 512x2048 (AR GEMV): %.3f ms/call  %.2f GMAC/s\n",
               dt / reps * 1e3, macs / dt / 1e9);
        ggml_gallocr_free(ga); ggml_backend_buffer_free(b); ggml_free(ctx);
    }

    // ---- (d/e) large GEMV: 64 MB f32 vs 16 MB q8, i.e. bigger than the 16 MB L3 ----
    // The small case above (4 MB f32) fits in cache, so it measures decode ALU work, not bytes.
    // The real AR streams an 84 MB weight file, so this case is the one that predicts it.
    {
        const int K = 512, N = 32768;
        for (int q = 0; q <= 1; q++) {
            struct ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(ip);
            struct ggml_tensor * w = ggml_new_tensor_2d(ctx, q ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, K, N);
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
            struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ggml_set_input(w); ggml_set_input(x); ggml_set_output(y);
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
            if (q) {
                float * src = (float *) malloc(sizeof(float) * K * N);
                for (int i = 0; i < K * N; i++) src[i] = (float) ((i * 37 % 97) - 48) / 48.0f;
                ggml_quantize_chunk(GGML_TYPE_Q8_0, src, (void *) w->data, 0, N, K, NULL);
                free(src);
            }
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, y);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            ggml_gallocr_alloc_graph(ga, g);
            ggml_backend_graph_compute(be, g);
            const int reps = 8;
            double t0 = now_s();
            for (int i = 0; i < reps; i++) ggml_backend_graph_compute(be, g);
            double dt = now_s() - t0;
            const double macs = (double) K * N * reps;
            const double bytes = (double) K * N * (q ? 1.0625 : 4.0);
            printf("large GEMV %-5s K=%d N=%d (%.0f MB): %.2f ms/call  %.2f GMAC/s  %.1f GB/s\n",
                   q ? "q8_0" : "f32", K, N, bytes / 1048576.0, dt / reps * 1e3, macs / dt / 1e9,
                   bytes * reps / dt / 1e9);
            ggml_gallocr_free(ga); ggml_backend_buffer_free(b); ggml_free(ctx);
        }
    }

    // ---- (f) bf16 GEMV: the vdpbf16ps path (bf16 weights, bf16 activations, fp32 accumulate) ----
    // Small (4 MB / 2 MB) and large (64 MB / 32 MB) so cache effects show up the same way as f32/q8.
    {
        const int Ns[2] = { 2048, 32768 };
        for (int sc = 0; sc < 2; sc++) {
            const int K = 512, N = Ns[sc];
            struct ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), NULL, true };
            struct ggml_context * ctx = ggml_init(ip);
            struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, N);
            struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
            struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ggml_set_input(w); ggml_set_input(x); ggml_set_output(y);
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
            {
                // bf16 with round-to-nearest-even, written straight into the tensor memory
                ggml_bf16_t * dst = (ggml_bf16_t *) w->data;
                for (int i = 0; i < K * N; i++) {
                    float v = (float) ((i * 37 % 97) - 48) / 48.0f;
                    uint32_t u;
                    memcpy(&u, &v, 4);
                    u += 0x7fffu + ((u >> 16) & 1u);
                    dst[i].bits = (uint16_t) (u >> 16);
                }
            }
            struct ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, y);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            ggml_gallocr_alloc_graph(ga, g);
            ggml_backend_graph_compute(be, g);
            const int reps = N <= 2048 ? 200 : 8;
            double t0 = now_s();
            for (int i = 0; i < reps; i++) ggml_backend_graph_compute(be, g);
            double dt = now_s() - t0;
            const double macs = (double) K * N * reps;
            const double bytes = (double) K * N * 2.0;
            printf("mul_mat bf16 %s K=%d N=%d: %.3f ms/call  %.2f GMAC/s  %.1f GB/s\n",
                   sc ? "large" : "small", K, N, dt / reps * 1e3, macs / dt / 1e9, bytes * reps / dt / 1e9);
            ggml_gallocr_free(ga); ggml_backend_buffer_free(b); ggml_free(ctx);
        }
    }

    ggml_backend_free(be);
    return 0;
}
