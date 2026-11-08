// 本地调试用的 AVX2 标量仿真垫片（只在 arm64 开发机上用，不进任何构建）。
//
// 目的：在没有任何 x86 CPU 消耗的前提下，把 AVX2 代码路径逐位等价地跑起来。
// FMA 用 fmaf() 模拟（单次舍入，与 vfmadd 同语义），shuffle 按 lane 展开，
// 因此语义与真机一致（不是“近似”）。用法：
//
//   clang++ -std=c++17 -O2 -D__AVX2__ -I tests/avx2_emu -I src <file>.cpp ...
//
// 注意：这是**opt-in**的 include 目录，绝不要把它加进正常构建路径。
#pragma once
#include <math.h>
#include <string.h>

typedef struct __m128 { float v[4]; } __m128;
typedef struct __m256 { float v[8]; } __m256;
typedef struct __m256i { int v[8]; } __m256i;
typedef struct __m256d { double v[4]; } __m256d;
typedef struct __m128i { long long v[2]; } __m128i;
typedef struct __m64 { float v[2]; } __m64;
enum { _CMP_LT_OQ = 0x11 };

static inline __m256 _mm256_setzero_ps() { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = 0.0f; return r; }
static inline __m256 _mm256_set1_ps(float a) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = a; return r; }
static inline __m256 _mm256_broadcast_ss(const float * p) { return _mm256_set1_ps(*p); }
static inline __m256 _mm256_loadu_ps(const float * p) { __m256 r; memcpy(r.v, p, 32); return r; }
static inline void _mm256_storeu_ps(float * p, __m256 a) { memcpy(p, a.v, 32); }
static inline __m256 _mm256_add_ps(__m256 a, __m256 b) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = a.v[i] + b.v[i]; return r; }
static inline __m256 _mm256_mul_ps(__m256 a, __m256 b) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = a.v[i] * b.v[i]; return r; }
static inline __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = fmaf(a.v[i], b.v[i], c.v[i]); return r; }
static inline __m256 _mm256_cmp_ps(__m256 a, __m256 b, int imm) { (void) imm; __m256 r; for (int i = 0; i < 8; i++) r.v[i] = (a.v[i] < b.v[i]) ? -1.0f : 0.0f; return r; }
static inline __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 m) {
    __m256 r;
    for (int i = 0; i < 8; i++) r.v[i] = (signbit(m.v[i]) && m.v[i] != 0.0f) ? b.v[i] : a.v[i];
    return r;
}

// 交错：unpacklo/hi_ps 以 128 位为界；unpacklo/hi_pd 以 64 位元素（= 2 个 float）为界
static inline __m256 _mm256_unpacklo_ps(__m256 a, __m256 b) {
    __m256 r;
    r.v[0] = a.v[0]; r.v[1] = b.v[0]; r.v[2] = a.v[1]; r.v[3] = b.v[1];
    r.v[4] = a.v[4]; r.v[5] = b.v[4]; r.v[6] = a.v[5]; r.v[7] = b.v[5];
    return r;
}
static inline __m256 _mm256_unpackhi_ps(__m256 a, __m256 b) {
    __m256 r;
    r.v[0] = a.v[2]; r.v[1] = b.v[2]; r.v[2] = a.v[3]; r.v[3] = b.v[3];
    r.v[4] = a.v[6]; r.v[5] = b.v[6]; r.v[6] = a.v[7]; r.v[7] = b.v[7];
    return r;
}
static inline __m256d _mm256_unpacklo_pd(__m256d a, __m256d b) {
    __m256d r;
    r.v[0] = a.v[0]; r.v[1] = b.v[0]; r.v[2] = a.v[2]; r.v[3] = b.v[2];
    return r;
}
static inline __m256d _mm256_unpackhi_pd(__m256d a, __m256d b) {
    __m256d r;
    r.v[0] = a.v[1]; r.v[1] = b.v[1]; r.v[2] = a.v[3]; r.v[3] = b.v[3];
    return r;
}
static inline __m256d _mm256_castps_pd(__m256 a) { __m256d r; memcpy(r.v, a.v, 32); return r; }
static inline __m256  _mm256_castpd_ps(__m256d a) { __m256 r; memcpy(r.v, a.v, 32); return r; }
// dst[127:0] = imm[0] ? a.hi : a.lo（imm[3] 置 0）；dst[255:128] = imm[4] ? b.hi : b.lo（imm[7] 置 0）
static inline __m256 _mm256_permute2f128_ps(__m256 a, __m256 b, int imm) {
    __m256 r;
    if (imm & 0x8) { r.v[0] = r.v[1] = r.v[2] = r.v[3] = 0.0f; }
    else if (imm & 0x1) { for (int i = 0; i < 4; i++) r.v[i] = a.v[i + 4]; }
    else { for (int i = 0; i < 4; i++) r.v[i] = a.v[i]; }
    if (imm & 0x80) { r.v[4] = r.v[5] = r.v[6] = r.v[7] = 0.0f; }
    else if (imm & 0x10) { for (int i = 0; i < 4; i++) r.v[i + 4] = b.v[i + 4]; }
    else { for (int i = 0; i < 4; i++) r.v[i + 4] = b.v[i]; }
    return r;
}

static inline __m256i _mm256_setr_epi32(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7) {
    __m256i r; r.v[0] = a0; r.v[1] = a1; r.v[2] = a2; r.v[3] = a3;
    r.v[4] = a4; r.v[5] = a5; r.v[6] = a6; r.v[7] = a7; return r;
}
static inline void _mm256_maskstore_ps(float * p, __m256i m, __m256 a) {
    for (int i = 0; i < 8; i++) if (m.v[i] < 0) p[i] = a.v[i];
}

static inline __m128 _mm256_castps256_ps128(__m256 a) {
    __m128 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i]; return r;
}
static inline __m128 _mm256_extractf128_ps(__m256 a, int hi) {
    __m128 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[(hi ? 4 : 0) + i]; return r;
}
static inline void _mm_storeu_ps(float * p, __m128 a) { memcpy(p, a.v, 16); }
static inline __m128 _mm_loadu_ps(const float * p) { __m128 r; memcpy(r.v, p, 16); return r; }
static inline __m128 _mm_add_ps(__m128 a, __m128 b) { __m128 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] + b.v[i]; return r; }
static inline __m128 _mm_mul_ps(__m128 a, __m128 b) { __m128 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i]; return r; }
static inline __m128 _mm_set1_ps(float a) { __m128 r; for (int i = 0; i < 4; i++) r.v[i] = a; return r; }
static inline __m128 _mm_setzero_ps(void) { __m128 r; for (int i = 0; i < 4; i++) r.v[i] = 0.0f; return r; }
static inline __m128 _mm_cmp_ps(__m128 a, __m128 b, int imm) { (void) imm; __m128 r; for (int i = 0; i < 4; i++) r.v[i] = (a.v[i] < b.v[i]) ? -1.0f : 0.0f; return r; }
static inline __m128 _mm_blendv_ps(__m128 a, __m128 b, __m128 m) {
    __m128 r;
    for (int i = 0; i < 4; i++) r.v[i] = (signbit(m.v[i]) && m.v[i] != 0.0f) ? b.v[i] : a.v[i];
    return r;
}
static inline __m128 _mm_castsi128_ps(__m128i a) { __m128 r; memcpy(r.v, &a, 16); return r; }
static inline __m128i _mm_castps_si128(__m128 a) { __m128i r; memcpy(&r, a.v, 16); return r; }
static inline __m128i _mm_loadl_epi64(const __m128i * p) { __m128i r; memcpy(&r, p, 8); return r; }
static inline void _mm_storel_epi64(__m128i * p, __m128i a) { memcpy(p, &a, 8); }
static inline __m128 _mm_castps128_256low(__m256 a) { return _mm256_castps256_ps128(a); }
static inline void _mm_storel_pi(__m64 * p, __m128 a) { p->v[0] = a.v[0]; p->v[1] = a.v[1]; }
