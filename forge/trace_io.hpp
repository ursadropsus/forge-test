// trace_io.hpp - conformance I/O for the Forge GPT-2 tap.
//
// Header-only, no dependencies beyond the C++17 standard library. The binary
// layout here must stay byte-for-byte identical to ../trace_format.py:
//
//   offset  0 : char   magic[8] = "FTRC0001"
//   offset  8 : uint32 layer
//   offset 12 : uint32 n_positions
//   offset 16 : uint32 d_mlp
//   offset 20 : uint32 dtype_code = 0 (float32 LE)
//   offset 24 : char   case_id[32] (NUL-padded)
//   offset 56 : float32 payload, row-major [position][neuron]
//
// Assumes a little-endian host with IEEE-754 float32; both are asserted below.
//
// Build the selftest:  g++ -std=c++17 -DTRACE_IO_SELFTEST -o trace_io_selftest trace_io.hpp

#ifndef FORGE_TRACE_IO_HPP
#define FORGE_TRACE_IO_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace conformance {

static_assert(sizeof(float) == 4, "float32 required");
static_assert(std::numeric_limits<float>::is_iec559, "IEEE-754 float required");

constexpr std::size_t kHeaderSize = 56;
constexpr std::uint32_t kDtypeF32LE = 0;

inline bool host_is_little_endian() {
    const std::uint32_t x = 1u;
    return *reinterpret_cast<const unsigned char*>(&x) == 1u;
}

// ---------------------------------------------------------------- writing

// data must point to n_positions * d_mlp contiguous floats, position-major.
inline void write_trace(const std::string& path,
                        const std::string& case_id,
                        std::uint32_t layer,
                        std::uint32_t n_positions,
                        std::uint32_t d_mlp,
                        const float* data) {
    if (!host_is_little_endian())
        throw std::runtime_error("trace_io: big-endian host not supported");
    if (case_id.size() > 32)
        throw std::runtime_error("trace_io: case_id longer than 32 chars: " + case_id);
    if (data == nullptr)
        throw std::runtime_error("trace_io: null data pointer for case " + case_id);

    unsigned char header[kHeaderSize];
    std::memset(header, 0, kHeaderSize);
    std::memcpy(header + 0, "FTRC0001", 8);
    const std::uint32_t fields[4] = {layer, n_positions, d_mlp, kDtypeF32LE};
    std::memcpy(header + 8, fields, sizeof(fields));
    std::memcpy(header + 24, case_id.data(), case_id.size());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("trace_io: cannot open for writing: " + path);
    out.write(reinterpret_cast<const char*>(header), kHeaderSize);
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(sizeof(float)) * n_positions * d_mlp);
    if (!out) throw std::runtime_error("trace_io: write failed: " + path);
    out.close();
}

// Convenience for a strided source: extracts [n_positions][d_mlp] where the
// value at (t, n) lives at data[t * row_stride + n]. Use when Forge's tensor
// is padded or interleaved rather than tightly packed.
inline void write_trace_strided(const std::string& path,
                                const std::string& case_id,
                                std::uint32_t layer,
                                std::uint32_t n_positions,
                                std::uint32_t d_mlp,
                                const float* data,
                                std::size_t row_stride) {
    std::vector<float> packed(static_cast<std::size_t>(n_positions) * d_mlp);
    for (std::uint32_t t = 0; t < n_positions; ++t)
        std::memcpy(packed.data() + static_cast<std::size_t>(t) * d_mlp,
                    data + t * row_stride, sizeof(float) * d_mlp);
    write_trace(path, case_id, layer, n_positions, d_mlp, packed.data());
}

// ---------------------------------------------------------------- reading

struct Case {
    std::string id;
    std::vector<int> token_ids;
};

// Parses tokens.tsv produced by export_reference.py:
//   <case_id>\t<n_tokens>\t<id,id,id,...>
// Lines beginning with '#' are comments.
inline std::vector<Case> read_token_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("trace_io: cannot open token file: " + path);
    std::vector<Case> cases;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t t1 = line.find('\t');
        const std::size_t t2 = (t1 == std::string::npos) ? std::string::npos
                                                         : line.find('\t', t1 + 1);
        if (t2 == std::string::npos)
            throw std::runtime_error("trace_io: malformed tokens.tsv at line "
                                     + std::to_string(lineno));
        Case c;
        c.id = line.substr(0, t1);
        const int declared = std::stoi(line.substr(t1 + 1, t2 - t1 - 1));
        std::stringstream ss(line.substr(t2 + 1));
        std::string field;
        while (std::getline(ss, field, ','))
            if (!field.empty()) c.token_ids.push_back(std::stoi(field));
        if (static_cast<int>(c.token_ids.size()) != declared)
            throw std::runtime_error("trace_io: case " + c.id + " declares "
                                     + std::to_string(declared) + " tokens but lists "
                                     + std::to_string(c.token_ids.size()));
        cases.push_back(std::move(c));
    }
    return cases;
}

// ------------------------------------------------------------- checksums

constexpr std::uint64_t kFnvOffset = 0xCBF29CE484222325ULL;
constexpr std::uint64_t kFnvPrime = 0x100000001B3ULL;

inline std::uint64_t fnv1a64(const void* data, std::size_t n_bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = kFnvOffset;
    for (std::size_t i = 0; i < n_bytes; ++i) { h ^= p[i]; h *= kFnvPrime; }
    return h;
}

// Orientation-invariant weight statistics, accumulated in double so that
// -ffast-math reassociation cannot change the answer materially. Compare these
// against weights_fingerprint.json to prove Forge loaded the same checkpoint.
struct TensorStats {
    std::size_t count = 0;
    double sum = 0.0, sumsq = 0.0;
    double min = 0.0, max = 0.0;
    std::uint64_t fnv1a64_layout = 0;
};

inline TensorStats tensor_stats(const float* data, std::size_t count) {
    TensorStats s;
    s.count = count;
    if (count == 0) return s;
    s.min = s.max = static_cast<double>(data[0]);
    for (std::size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(data[i]);
        s.sum += v;
        s.sumsq += v * v;
        if (v < s.min) s.min = v;
        if (v > s.max) s.max = v;
    }
    s.fnv1a64_layout = fnv1a64(data, count * sizeof(float));
    return s;
}

inline void print_tensor_stats(const std::string& name, const TensorStats& s) {
    std::printf("%-40s count=%zu sum=%.10g sumsq=%.10g min=%.10g max=%.10g fnv=%016llx\n",
                name.c_str(), s.count, s.sum, s.sumsq, s.min, s.max,
                static_cast<unsigned long long>(s.fnv1a64_layout));
}

}  // namespace conformance

#ifdef TRACE_IO_SELFTEST
#include <cassert>
int main() {
    using namespace conformance;
    assert(host_is_little_endian());
    const char probe[] = "conformance";
    const std::uint64_t h = fnv1a64(probe, 11);
    std::printf("fnv1a64('conformance') = %016llx\n",
                static_cast<unsigned long long>(h));
    if (h != 0xa947cd7007eb20f2ULL) {
        std::printf("MISMATCH with trace_format.py\n");
        return 1;
    }
    std::vector<float> a(6 * 3072);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) / 7.0f;
    write_trace("/tmp/cxx_roundtrip_L05.ftrc", "roundtrip", 5, 6, 3072, a.data());
    std::printf("wrote /tmp/cxx_roundtrip_L05.ftrc (%zu bytes expected)\n",
                kHeaderSize + a.size() * 4);
    return 0;
}
#endif
#endif  // FORGE_TRACE_IO_HPP
