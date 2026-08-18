# Forge / Hugging Face activation conformance

A small test package for [Forge](https://github.com/muchlakshay/Forge). It checks
whether Forge's GPT-2 and Hugging Face `transformers` compute the *same internal
activations*, not just the same output tokens.

Everything needed to run it is in this repo, including the reference traces. No
account, no corpus, no dependencies beyond what Forge already needs.

---

## The ask

Run one executable on 31 frozen token sequences and send back the folder it
writes. If you have Forge building already, this is about twenty minutes,
most of which is a rebuild.

If it looks like a waste of your time, it probably is — I'm a non-expert working with AI, and may have missed something obvious.
I'd rather you make that call than have me guess.

---

## Why you might want this

Forge's README says its GPT-2 matches Hugging Face token-for-token under greedy
decoding. That's an output-level claim, and it's a good one.

This tests something stricter. It compares the **post-GELU MLP activation at
layer 5** — a 3,072-wide vector at every token position — between Forge and a
pinned Hugging Face stack, across 31 inputs. Output-level agreement can survive
a compensating error somewhere in the middle. Activation-level agreement is much
harder to get by accident.

So if it passes, you have a stronger claim than the one you currently make. If
it fails, you get the specific case, token position, and neuron index where the
two implementations first diverge — which is usually the hard part of debugging
a numerical difference.

The comparison script is small and has no dependencies beyond numpy. If it's
useful, keep it; it would work as a CI check.

## Why I want this

I do interpretability work on GPT-2 Small — specifically, characterising what
individual neurons in layer 5 respond to. One of them, neuron 541, responds to
completed paired constructions: `up and down`, `now and then`, `here and there`,
`top to bottom`. It peaks on the final token of the pair.

All of that is measured on one software stack. A finding that only exists in
PyTorch is weaker than one that reproduces in an independent implementation.
Forge is the only from-scratch GPT-2 I found that loads the real pretrained
weights and is complete enough to test this against — which is why I'm asking
you rather than someone else.

---

## What's in here

```
fixture/
  fixture.json              31 cases: exact token IDs, checkpoint identity, env record
  tokens.tsv                the same token IDs, one line per case (what Forge reads)
  weights_fingerprint.json  per-tensor checksums to confirm the same weights loaded
  traces/cpu/*.ftrc         PyTorch CPU reference activations
  traces/cuda/*.ftrc        PyTorch CUDA reference activations
cases.json                  the 31 inputs as plain text, if you want to read them
config.json                 frozen conventions and the pass criteria
forge/
  conformance.cpp           the driver to build against Forge
  trace_tap.hpp             the activation capture point
  trace_io.hpp              binary trace format + checksums (header-only)
  PATCH_NOTES.md            install and build detail
compare_traces.py           the comparison (numpy only)
inspect_traces.py           per-case summary of one trace set
trace_format.py             Python side of the binary format
```

## Running it

Full detail is in `forge/PATCH_NOTES.md`. The short version:

1. Copy the three files from `forge/` into `experiments/GPT-2/`.
2. Add two lines to `experiments/CMakeLists.txt`:
   ```cmake
   add_executable(conformance GPT-2/conformance.cpp)
   target_link_libraries(conformance PRIVATE Forge)
   ```
   (On Windows, also add `conformance` to the `foreach (target gpt2 mnist)` list
   so the OpenBLAS DLL gets copied next to it.)
3. Build, then:
   ```bash
   export OPENBLAS_NUM_THREADS=1
   ./build/conformance --tokens fixture/tokens.tsv --weights model.safetensors \
       --out fixture/traces/forge --layers 5
   ```
4. Send back `fixture/traces/forge/`. That's it — I'll run the comparison.

If you want to run the comparison yourself: `python compare_traces.py --fixture
fixture --ref cpu --alt-ref cuda --test fixture/traces/forge --out report`.

**`gpt2.cpp` is not modified.** `conformance.cpp` is a separate executable that
copies your model definition verbatim and adds two marked lines: one call after
the GELU to capture the activation, and an indexed block loop so the capture
knows which layer it's in. Your working program is untouched.

---

## Things you should know before spending time on it

**The driver compiles but has never run.** I don't have a machine that can build
Forge. `conformance.cpp` compiles clean against your headers with your flags,
which means the API calls are right, but that's all it means. If the first run
fails on a shape assertion or a bad flag, that's my bug, not Forge's.

**Two things need pinning or the comparison is meaningless.**

*Weights.* The reference is pinned to revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` of `openai-community/gpt2`. Forge
downloads from the moving `main` URL. These may currently be identical, but the
comparison shouldn't rest on that. `download_GPT2_124M` skips the download if
`./model.safetensors` already exists, so dropping the pinned file in place is
enough — no code change. `--dump-weight-stats` prints checksums to confirm.

*Threading.* The driver defaults to `--threads 1`, and you also need
`OPENBLAS_NUM_THREADS=1`. Eigen's thread pool and OpenBLAS both change the order
floating-point sums accumulate in, which changes the last few bits. Worth
running twice and diffing the output files before trusting anything.

**Tolerance is measured, not chosen.** The package ships both a CPU and a CUDA
PyTorch reference. The gap between them — 2.4e-06 to 1.0e-05 on these cases — is
what identical code costs when you change backend. Forge is judged against that
envelope rather than a threshold I picked. Bit-identical agreement is not the
expectation and not the pass condition.

**Not a pass/fail on max error.** The criteria are: same peak position, same
winning neuron, top-20 neuron overlap, ranking correlation, and no sign flip on
the margin. Raw numerical difference is reported but doesn't fail anything, and
close calls are reported as INDETERMINATE rather than as agreement or
disagreement.

---

## One question

The shipped `experiments/GPT-2/gpt2.cpp` defines `argmax` but doesn't use it —
generation goes through `sample_top_k(pred, 50)`. So the repo as it stands
doesn't itself demonstrate the greedy token-for-token claim in the README.

Did you test that separately, or in a version that isn't checked in? Not a
gotcha — I'd just like to cite it accurately, and if the check exists it's worth
having in the repo.

(This package doesn't touch it either way: it runs a single forward pass per
case and never generates, so sampling is out of the picture entirely.)

---

Licence: same as Forge (MIT), do whatever you like with it. Happy to be reached
wherever this was posted.
