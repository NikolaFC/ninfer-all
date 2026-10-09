// GGUF Q2_0 and Q8_0 CPU expert rows against FP64 from independently decoded stored blocks, at the
// Flash-Next expert shapes, for every backend this host supports.
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

std::uint16_t half_bits(float value) {
    // Positive normal values only, rounded to nearest by truncating a rounded mantissa.
    const auto word = std::bit_cast<std::uint32_t>(value);
    const int exponent = int((word >> 23) & 255) - 127 + 15;
    const std::uint32_t mantissa = ((word & 0x7fffff) + 0x1000) >> 13;
    return static_cast<std::uint16_t>((exponent << 10) + mantissa);
}

std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    return static_cast<std::uint16_t>(word >> 16);
}

struct Matrix {
    QType format;
    int rows, k;
    std::int64_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<double> values; // decoded independently from the stored bytes
};

Matrix make(QType format, int rows, int k, std::mt19937& rng) {
    const int group = format == QType::GGUF_Q2_0 ? 64 : 32;
    const int block = format == QType::GGUF_Q2_0 ? 18 : 34;
    Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}};
    m.bytes.resize(std::size_t(rows) * m.row_bytes);
    std::uniform_real_distribution<float> scale(0.002F, 0.05F);
    for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            std::uint8_t* p     = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const auto d        = half_bits(scale(rng));
            const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
            std::memcpy(p, &signed_d, 2);
        }
    }
    m.values.resize(std::size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            const std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const double d        = exact_half(unsigned(p[0]) | (unsigned(p[1]) << 8));
            for (int j = 0; j < group; ++j) {
                const double code = format == QType::GGUF_Q2_0
                                        ? double(((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1)
                                        : double(static_cast<std::int8_t>(p[2 + j]));
                m.values[std::size_t(r) * k + std::size_t(b) * group + j] = d * code;
            }
        }
    }
    return m;
}

void check(const Matrix& m, int tokens, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(std::size_t(tokens) * m.k);
    for (auto& v : x) { v = bf16(normal(rng)); }
    std::vector<double> oracle(std::size_t(tokens) * m.rows);
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < m.rows; ++r) {
            double sum = 0;
            for (int i = 0; i < m.k; ++i) {
                sum += m.values[std::size_t(r) * m.k + i] *
                       double(std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + i]) << 16));
            }
            oracle[std::size_t(t) * m.rows + r] = sum;
        }
    }
    GgufCpuActivation a(16, m.k);
    a.prepare_bf16(m.format, x.data(), m.k, tokens);
    const GgufCpuMatrix w{m.format, m.bytes.data(), m.row_bytes, m.rows, m.k};
    std::vector<float> scalar(oracle.size());
    gguf_cpu_rows(w, a, 0, m.rows, scalar.data(), m.rows, GgufCpuBackend::Scalar);
    for (const auto backend : {GgufCpuBackend::Scalar, GgufCpuBackend::Avx2, GgufCpuBackend::Avx512Vnni}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        std::vector<float> y(oracle.size()), again(oracle.size());
        // Two row ranges, as workers split them.
        gguf_cpu_rows(w, a, 0, m.rows / 3, y.data(), m.rows, backend);
        gguf_cpu_rows(w, a, m.rows / 3, m.rows, y.data(), m.rows, backend);
        gguf_cpu_rows(w, a, 0, m.rows, again.data(), m.rows, backend);
        double error = 0, norm = 0, peak = 0, worst = 0, drift = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            require(std::isfinite(y[i]), "nonfinite output");
            require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(again[i]),
                    "a row split changed the result or a repeat differed");
            error += (double(y[i]) - oracle[i]) * (double(y[i]) - oracle[i]);
            norm += oracle[i] * oracle[i];
            peak  = std::max(peak, std::abs(oracle[i]));
            worst = std::max(worst, std::abs(double(y[i]) - oracle[i]));
            drift = std::max(drift, std::abs(double(y[i]) - double(scalar[i])));
        }
        const double relative = std::sqrt(error / norm);
        std::cout << gguf_cpu_backend_name(backend) << " format " << int(m.format) << " "
                  << m.rows << "x" << m.k << " T=" << tokens << ": relative L2 " << relative
                  << ", max error/peak " << worst / peak << ", vs scalar " << drift / peak << '\n';
        require(relative < 0.03 && worst / peak < 0.1, "error exceeds the int8-activation bound");
        require(drift / peak < 1e-5, "vector backend disagrees with the scalar arithmetic");
    }
}

} // namespace

int main() {
    try {
        std::mt19937 rng(20261009);
        for (const QType format : {QType::GGUF_Q2_0, QType::GGUF_Q8_0}) {
            const Matrix gate = make(format, 640, 2560, rng);
            const Matrix down = make(format, 2560, 640, rng);
            for (const int tokens : {1, 2, 3, 4, 5, 8, 9}) {
                check(gate, tokens, rng);
                check(down, tokens, rng);
            }
        }
        bool refused = false;
        try {
            GgufCpuActivation a(1, 2560);
            std::vector<std::uint16_t> x(2560 * 2);
            a.prepare_bf16(QType::GGUF_Q2_0, x.data(), 2560, 2);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "an activation above its capacity was accepted");
        std::cout << "gguf_expert_cpu: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_expert_cpu: " << error.what() << '\n';
        return 1;
    }
}
