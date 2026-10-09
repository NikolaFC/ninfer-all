#pragma once

// Host-resident GGUF routed experts computed on the CPU, from the same stored blocks the GPU reads
// (Q2_0 and Q8_0 today). The Program splits a layer's missing experts between copies to the GPU
// and these products; it owns the threads, the activations and the weighted merge.
//
// Arithmetic: the activation is quantized per stored block to int8 with an FP32 scale (64 values
// for Q2_0, 32 for Q8_0), each block's integer dot product is exact, and the block products are
// summed in FP32. The middle silu(gate . x) * (up . x) stays FP32 and is quantized the same way for
// the down product. This is not the GPU's q8_1 arithmetic, so a pair computed here can differ from
// the GPU's in the last bits; the independent FP64 oracle bounds both.

#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::ops {

enum class GgufCpuBackend { Automatic, Scalar, Avx2, Avx512Vnni };
[[nodiscard]] bool gguf_cpu_backend_available(GgufCpuBackend backend) noexcept;
[[nodiscard]] const char* gguf_cpu_backend_name(GgufCpuBackend backend) noexcept;
[[nodiscard]] GgufCpuBackend gguf_cpu_backend_selected(GgufCpuBackend requested) noexcept;
// Whether `format` has a CPU product here.
[[nodiscard]] bool gguf_cpu_supported(QType format) noexcept;

// One stored projection: `rows` rows of `k` values, each row `row_bytes` of GGUF blocks.
struct GgufCpuMatrix {
    QType format             = QType::GGUF_Q2_0;
    const std::uint8_t* data = nullptr;
    std::int64_t row_bytes   = 0;
    int rows                 = 0;
    int k                    = 0;
};

// Activations of up to `tokens` columns of `k` values, quantized for one block format.
class GgufCpuActivation {
public:
    GgufCpuActivation() = default;
    GgufCpuActivation(int token_capacity, int k);
    // Quantizes `tokens` FP32 columns of k values (column-major, `stride` apart) for `format`.
    void prepare(QType format, const float* values, std::size_t stride, int tokens);
    // The same from BF16 words.
    void prepare_bf16(QType format, const std::uint16_t* values, std::size_t stride, int tokens);
    [[nodiscard]] int tokens() const noexcept { return tokens_; }
    [[nodiscard]] int k() const noexcept { return k_; }
    [[nodiscard]] QType format() const noexcept { return format_; }
    [[nodiscard]] const std::int8_t* codes(int token) const noexcept {
        return codes_.data() + std::size_t(token) * k_;
    }
    [[nodiscard]] const float* scales(int token) const noexcept {
        return scales_.data() + std::size_t(token) * groups_;
    }
    [[nodiscard]] const std::int32_t* sums(int token) const noexcept {
        return sums_.data() + std::size_t(token) * groups_;
    }

private:
    void quantize(QType format, int token, const float* values);
    int capacity_ = 0, k_ = 0, groups_ = 0, tokens_ = 0;
    QType format_ = QType::GGUF_Q2_0;
    std::vector<std::int8_t> codes_;
    std::vector<float> scales_, scratch_;
    std::vector<std::int32_t> sums_;
};

// y[t * y_stride + r] = row r . x[t] for rows [row_begin, row_end) and every prepared token.
void gguf_cpu_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                   float* y, std::size_t y_stride, GgufCpuBackend backend);

namespace gguf_cpu_detail {
using Rows = void (*)(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*,
                      std::size_t);
void rows_scalar(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*, std::size_t);
#if defined(NINFER_CPU_EXPERT_X86)
void rows_avx2(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*, std::size_t);
void rows_avx512(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*, std::size_t);
#endif
// Q2_0 activations are stored in the lane order of the vector decoders: within each 64-value
// block, position L * 16 + i holds value 4 * i + L.
inline int q2_lane_position(int j) noexcept { return (j % 4) * 16 + j / 4; }
float half_to_float(std::uint16_t bits) noexcept;
} // namespace gguf_cpu_detail

} // namespace ninfer::ops
