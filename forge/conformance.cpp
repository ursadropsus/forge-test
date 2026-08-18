// conformance.cpp - trace-only driver for Forge's GPT-2.
//
// Drop this into experiments/GPT-2/ alongside trace_io.hpp and trace_tap.hpp,
// then add one line to experiments/CMakeLists.txt (see PATCH_NOTES.md).
// gpt2.cpp is not modified.
//
// The model definition below is copied verbatim from experiments/GPT-2/gpt2.cpp
// with exactly two additions, both marked CONFORMANCE:
//   1. TransformerBlock::operator() calls tap::capture() after the GELU
//   2. GPT2::operator() sets tap::current_layer from its block loop index
//
// Everything else - layer construction, weight loading, transposes, positional
// encoding - is unchanged, so a divergence found here is a divergence in the
// same code path the real gpt2 executable runs.
//
// What this deliberately does NOT do:
//   * no tokenisation - token ids come from the fixture, so Forge's BPE and the
//     missing BOS are taken out of the comparison entirely
//   * no generation - one forward pass per case, so sample_top_k never runs and
//     the unverified greedy-decoding claim is irrelevant to these traces
//   * no weight download - the pinned model.safetensors must already exist

#include "include/Forge.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "trace_io.hpp"
#include "trace_tap.hpp"

using namespace Forge;

// ---------------------------------------------------------------- copied from gpt2.cpp

void transpose(Tensor& A, Tensor& B, const std::vector<int>& perm) {
    if (A.shape().size() != B.shape().size()) throw std::invalid_argument("Different shapes");
    forge_transpose_AVX2<float>(static_cast<float*>(A.data()),
        static_cast<float*>(B.data()), A.strides(), A.shape(), perm, A.size());
}

static constexpr std::size_t D_MODEL{768}, QK_DIMS{64}, V_DIMS{64};
static constexpr std::size_t HEADS{12}, VOCAB_SIZE{50257}, MAX_SEQ_LEN{1024};

struct TransformerBlock {
    LayerNorm m_ln1{D_MODEL, false};
    SelfAttention m_attn1{D_MODEL, QK_DIMS, V_DIMS, HEADS, true, false, true, true};
    LayerNorm m_ln2{D_MODEL, false};
    Linear m_l1{D_MODEL, 3072, false};
    Gelu m_gelu{};
    Linear m_l2{3072, D_MODEL, false};

    Tensor operator()(Tensor& x) {
        auto ln1 {m_ln1(x)};
        auto attn1 {m_attn1(ln1)};
        auto x2 {x.BroadcastAdd(attn1)};

        auto ln2 {m_ln2(x2)};
        auto l1 {m_l1(ln2)};
        auto gelu {m_gelu(l1)};

        // CONFORMANCE: the only addition to the forward path.
        if (conformance::tap::enabled) {
            const auto& shp {gelu.shape()};
            std::uint32_t n_pos{}, d{};
            if (shp.size() == 2) {
                n_pos = static_cast<std::uint32_t>(shp[0]);
                d     = static_cast<std::uint32_t>(shp[1]);
            } else if (shp.size() == 3 && shp[0] == 1) {
                n_pos = static_cast<std::uint32_t>(shp[1]);
                d     = static_cast<std::uint32_t>(shp[2]);
            } else {
                throw std::runtime_error("conformance: unexpected gelu rank "
                                         + std::to_string(shp.size()));
            }
            conformance::tap::capture(static_cast<const float*>(gelu.data()), n_pos, d);
        }

        auto l2 {m_l2(gelu)};
        auto opt {l2.BroadcastAdd(x2)};
        return opt;
    }
};

struct PosEnc {Tensor m_pe;};
struct Members {
    Embedding m_embed{D_MODEL, VOCAB_SIZE, Dtype::float32, false};
    PosEnc m_pe;
    std::array<TransformerBlock, 12> m_blocks{};
    LayerNorm m_ln{D_MODEL, false};
    Linear m_l_opt{1, 1, false, false};
};

struct GPT2 {
    Members m_mem;
    Tensor operator()(Tensor& x) {
        auto seq_len {x.size()};
        auto PE {Tensor::FromHostPtr(static_cast<float*>(m_mem.m_pe.m_pe.data()), {seq_len, D_MODEL}, false)};

        if (auto& mask {m_mem.m_blocks[0].m_attn1.mask()}; mask.size()==0 || mask.shape().back()!=seq_len ) {
            for (auto& block:m_mem.m_blocks) block.m_attn1.createMask(seq_len);
        }

        auto embd {m_mem.m_embed(x)};
        auto pe_added {embd.BroadcastAdd(PE)};
        auto block_opt {pe_added};

        // CONFORMANCE: indexed loop so the tap knows which block it is in.
        for (std::size_t i{}; i < m_mem.m_blocks.size(); ++i) {
            conformance::tap::current_layer = static_cast<int>(i);
            block_opt = m_mem.m_blocks[i](block_opt);
        }
        conformance::tap::current_layer = -1;

        auto ln {m_mem.m_ln(block_opt)};
        auto opt {m_mem.m_l_opt(ln)};
        return opt;
    }
};

void load_gpt2(Members& mem, const std::string& filename) {
    auto gpt_state_dict {load_safetensors(filename)};
    mem.m_embed.all_embeddings() = gpt_state_dict["wte.weight"];
    mem.m_pe.m_pe = gpt_state_dict["wpe.weight"];
    mem.m_ln.gamma() = gpt_state_dict["ln_f.weight"];
    mem.m_ln.beta() = gpt_state_dict["ln_f.bias"];
    mem.m_l_opt.weights() = gpt_state_dict["wte.weight"];
    mem.m_l_opt.set_shape(D_MODEL, VOCAB_SIZE);

    for (int i {}; i<12; ++i) {
        auto& block {mem.m_blocks[i]};
        block.m_ln1.gamma() = gpt_state_dict[std::format("h.{}.ln_1.weight", i)];
        block.m_ln1.beta() = gpt_state_dict[std::format("h.{}.ln_1.bias", i)];
        block.m_ln2.gamma() = gpt_state_dict[std::format("h.{}.ln_2.weight", i)];
        block.m_ln2.beta() = gpt_state_dict[std::format("h.{}.ln_2.bias", i)];

        transpose(gpt_state_dict[std::format("h.{}.mlp.c_fc.weight", i)], block.m_l1.weights(), {1, 0});
        block.m_l1.bias() = gpt_state_dict[std::format("h.{}.mlp.c_fc.bias", i)];

        transpose(gpt_state_dict[std::format("h.{}.mlp.c_proj.weight", i)], block.m_l2.weights(), {1, 0});
        block.m_l2.bias() = gpt_state_dict[std::format("h.{}.mlp.c_proj.bias", i)];

        auto& attn_w {gpt_state_dict[std::format("h.{}.attn.c_attn.weight", i)]};
        Tensor qkv_t {{D_MODEL*3, D_MODEL}, Dtype::float32, false};
        transpose(attn_w, qkv_t, {1, 0});

        std::size_t component_size {D_MODEL*D_MODEL};

        auto* ptr {static_cast<float*>(qkv_t.data())};
        auto qw {Tensor::FromHostPtr(ptr, {12, 64, 768}, false).clone()};
        auto kw {Tensor::FromHostPtr(ptr+component_size, {12, 64, 768}, false).clone()};
        auto vw {Tensor::FromHostPtr(ptr+2*component_size, {12, 64, 768}, false).clone()};

        transpose(qw,block.m_attn1.query(), {0, 2, 1});
        transpose(kw,block.m_attn1.key(), {0, 2, 1});
        transpose(vw,block.m_attn1.value(), {0, 2, 1});

        auto* bptr {static_cast<float*>(gpt_state_dict[std::format("h.{}.attn.c_attn.bias", i)].data())};
        block.m_attn1.query_bias() = Tensor::FromHostPtr(bptr, {HEADS, QK_DIMS}, false).clone();
        block.m_attn1.key_bias() = Tensor::FromHostPtr(bptr+D_MODEL, {HEADS, QK_DIMS}, false).clone();
        block.m_attn1.value_bias() = Tensor::FromHostPtr(bptr+2*D_MODEL, {HEADS, QK_DIMS}, false).clone();

        transpose(gpt_state_dict[std::format("h.{}.attn.c_proj.weight", i)], block.m_attn1.linear().weights(), {1, 0});
        block.m_attn1.linear().bias() = gpt_state_dict[std::format("h.{}.attn.c_proj.bias", i)];
    }
}

// ---------------------------------------------------------------- driver

namespace {

struct Args {
    std::string tokens, weights{"./model.safetensors"}, out{"./traces/forge"};
    std::vector<int> layers{5};
    int threads{1};
    bool dump_weight_stats{false};
};

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> v;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string piece = s.substr(start, comma - start);
        if (!piece.empty()) v.push_back(std::stoi(piece));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return v;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if      (k == "--tokens")  a.tokens  = next();
        else if (k == "--weights") a.weights = next();
        else if (k == "--out")     a.out     = next();
        else if (k == "--layers")  a.layers  = parse_int_list(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--dump-weight-stats") a.dump_weight_stats = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", k.c_str()); std::exit(2); }
    }
    if (a.tokens.empty()) {
        std::fprintf(stderr,
            "usage: %s --tokens tokens.tsv [--weights model.safetensors]\n"
            "          [--out DIR] [--layers 5] [--threads 1] [--dump-weight-stats]\n\n"
            "  --threads 1 (default) forces Forge's Eigen thread pool single-threaded.\n"
            "  Also set OPENBLAS_NUM_THREADS=1 in the environment - Eigen hands GEMMs\n"
            "  to OpenBLAS, whose own threading is not controlled by --threads.\n",
            argv[0]);
        std::exit(2);
    }
    return a;
}

void dump_stats(const std::string& name, const Tensor& t) {
    conformance::print_tensor_stats(
        name, conformance::tensor_stats(static_cast<const float*>(t.data()), t.size()));
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // Single-threaded by default: Eigen's thread pool splits reductions across
    // threads, and the split can change run to run. Determinism first; measure
    // the cost of threading afterwards by re-running with --threads N.
    global_exeCtx.set_threads(args.threads);
    std::printf("threads: %d (OPENBLAS_NUM_THREADS=%s)\n", args.threads,
                std::getenv("OPENBLAS_NUM_THREADS") ? std::getenv("OPENBLAS_NUM_THREADS") : "unset");

    if (!std::filesystem::exists(args.weights)) {
        std::fprintf(stderr,
            "ERROR: %s not found.\n"
            "This driver never downloads: the checkpoint must be the pinned revision\n"
            "the Python exporter hashed, not whatever HEAD happens to be today.\n",
            args.weights.c_str());
        return 2;
    }

    const auto cases = conformance::read_token_file(args.tokens);
    std::printf("loaded %zu cases from %s\n", cases.size(), args.tokens.c_str());

    GPT2 gpt2;
    load_gpt2(gpt2.m_mem, args.weights);
    std::printf("loaded weights from %s\n", args.weights.c_str());

    if (args.dump_weight_stats) {
        // Compare against weights_fingerprint.json. sum/sumsq/min/max are computed
        // in double on both sides and are invariant to storage order, so they hold
        // even though Forge transposes c_fc/c_proj out of Hugging Face's Conv1D
        // layout at load time. The fnv hash will differ for exactly that reason -
        // matching invariants with a differing hash means "same weights, different
        // layout", which is expected here and worth recording.
        std::printf("\n--- weight stats (compare to weights_fingerprint.json) ---\n");
        dump_stats("transformer.wte.weight", gpt2.m_mem.m_embed.all_embeddings());
        dump_stats("transformer.wpe.weight", gpt2.m_mem.m_pe.m_pe);
        for (int L : args.layers) {
            auto& b = gpt2.m_mem.m_blocks[L];
            dump_stats(std::format("transformer.h.{}.ln_2.weight", L), b.m_ln2.gamma());
            dump_stats(std::format("transformer.h.{}.ln_2.bias", L), b.m_ln2.beta());
            dump_stats(std::format("transformer.h.{}.mlp.c_fc.weight", L), b.m_l1.weights());
            dump_stats(std::format("transformer.h.{}.mlp.c_fc.bias", L), b.m_l1.bias());
            dump_stats(std::format("transformer.h.{}.mlp.c_proj.weight", L), b.m_l2.weights());
            dump_stats(std::format("transformer.h.{}.mlp.c_proj.bias", L), b.m_l2.bias());
        }
        std::printf("--- end weight stats ---\n\n");
    }

    std::filesystem::create_directories(args.out);

    conformance::tap::enabled = true;
    conformance::tap::capture_layers.clear();
    for (int L : args.layers) {
        if (L < 0 || L > 11) { std::fprintf(stderr, "layer %d out of range 0-11\n", L); return 2; }
        conformance::tap::capture_layers.insert(L);
    }

    int written = 0;
    for (const auto& c : cases) {
        conformance::tap::begin_pass();

        std::vector<int> ids {c.token_ids};   // FromHostPtr wants a non-const pointer
        auto ids_t {Tensor::FromHostPtr(ids.data(), {ids.size()}, false)};
        auto logits {gpt2(ids_t)};            // one forward pass, no generation
        (void)logits;

        for (int L : args.layers) {
            if (!conformance::tap::have(L)) {
                std::fprintf(stderr, "case %s: nothing captured for layer %d\n", c.id.c_str(), L);
                return 1;
            }
            const auto n_pos = conformance::tap::n_positions[L];
            const auto d     = conformance::tap::d_mlp[L];
            if (n_pos != ids.size()) {
                std::fprintf(stderr,
                    "case %s: captured %u positions but the fixture has %zu tokens.\n",
                    c.id.c_str(), n_pos, ids.size());
                return 1;
            }
            if (d != 3072) {
                std::fprintf(stderr, "case %s: d_mlp is %u, expected 3072\n", c.id.c_str(), d);
                return 1;
            }
            const auto path = std::format("{}/{}_L{:02d}.ftrc", args.out, c.id, L);
            conformance::write_trace(path, c.id, static_cast<std::uint32_t>(L),
                                     n_pos, d, conformance::tap::buffers[L].data());
            ++written;
        }
        std::printf("  %-6s %4zu tokens -> ok\n", c.id.c_str(), ids.size());
    }

    std::printf("wrote %d traces to %s\n", written, args.out.c_str());
    return 0;
}
