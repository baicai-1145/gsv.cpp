// 本地调试用的 AVX2 标量仿真垫片（只在 arm64 开发机上用，不进构建）。
// 目的：在没有任何 x86 CPU 消耗的前提下，把 gsv_kernels.cpp 的 *AVX2 路径*
// 逐位等价地跑起来。FMA 用 fmaf() 模拟（单次舍入，与 vfmadd 同语义），
// 比较/选择按 lane 展开，因此语义与真机一致（不是"近似"）。
#pragma once
#include <cmath>
#include <cstring>

struct __m256 { float v[8]; };
enum { _CMP_LT_OQ = 0x11 };

static inline __m256 _mm256_setzero_ps() { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = 0.0f; return r; }
static inline __m256 _mm256_set1_ps(float a) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = a; return r; }
static inline __m256 _mm256_broadcast_ss(const float * p) { return _mm256_set1_ps(*p); }
static inline __m256 _mm256_loadu_ps(const float * p) { __m256 r; std::memcpy(r.v, p, 32); return r; }
static inline void _mm256_storeu_ps(float * p, __m256 a) { std::memcpy(p, a.v, 32); }
static inline __m256 _mm256_add_ps(__m256 a, __m256 b) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = a.v[i] + b.v[i]; return r; }
static inline __m256 _mm256_mul_ps(__m256 a, __m256 b) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = a.v[i] * b.v[i]; return r; }
static inline __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = std::fmaf(a.v[i], b.v[i], c.v[i]); return r; }
static inline __m256 _mm256_cmp_ps(__m256 a, __m256 b, int) { __m256 r; for (int i = 0; i < 8; i++) r.v[i] = (a.v[i] < b.v[i]) ? -1.0f : 0.0f; return r; }
static inline __m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 m) {
    __m256 r;
    for (int i = 0; i < 8; i++) r.v[i] = (std::signbit(m.v[i]) && m.v[i] != 0.0f) ? b.v[i] : a.v[i];
    return r;
}
