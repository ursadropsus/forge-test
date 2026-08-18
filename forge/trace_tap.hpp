// trace_tap.hpp - capture point for Forge's post-GELU MLP activation.
//
// The layer index is set explicitly by the model's block loop rather than
// inferred by counting calls. Forge's GPT2::operator() iterates
// std::array<TransformerBlock, 12> in order, so the index is already available
// at the call site - there is no reason to guess it.
//
// No Forge headers are included here on purpose: this file only deals in raw
// float pointers and sizes, so it compiles and tests standalone.

#ifndef FORGE_TRACE_TAP_HPP
#define FORGE_TRACE_TAP_HPP

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace conformance {
namespace tap {

inline bool enabled = false;
inline std::set<int> capture_layers;               // block indices to keep
inline int current_layer = -1;                     // set by the block loop
inline std::map<int, std::vector<float>> buffers;  // layer -> [pos][neuron]
inline std::map<int, std::uint32_t> n_positions;   // layer -> seq_len
inline std::map<int, std::uint32_t> d_mlp;         // layer -> 3072

// Call once immediately before each forward pass.
inline void begin_pass() {
    current_layer = -1;
    buffers.clear();
    n_positions.clear();
    d_mlp.clear();
}

// Call from inside TransformerBlock::operator(), immediately after the GELU.
// `data` points to n_pos * d contiguous floats, position-major. Forge's GPT-2
// runs without a batch dimension, so the gelu tensor is exactly (seq_len, 3072)
// and this is a direct view of it.
inline void capture(const float* data, std::uint32_t n_pos, std::uint32_t d) {
    if (!enabled) return;
    if (current_layer < 0) return;
    if (capture_layers.find(current_layer) == capture_layers.end()) return;
    if (data == nullptr)
        throw std::runtime_error("trace_tap: null activation pointer at layer "
                                 + std::to_string(current_layer));
    buffers[current_layer].assign(data, data + static_cast<std::size_t>(n_pos) * d);
    n_positions[current_layer] = n_pos;
    d_mlp[current_layer] = d;
}

inline bool have(int layer) { return buffers.find(layer) != buffers.end(); }

}  // namespace tap
}  // namespace conformance

#endif  // FORGE_TRACE_TAP_HPP
