# Installing the tap into Forge

Written against the actual repository (`muchlakshay/Forge`, `experiments/GPT-2/gpt2.cpp`),
not against guesses. `conformance.cpp` has been compiled against Forge's real
headers with Forge's own flags and produces zero errors — so the API calls are
correct. It has **not been executed**, because building the full library needs
more machine than I have here, and I have no GPT-2 weights.

## What gets changed

Nothing in `gpt2.cpp`. The conformance driver is a **separate executable** that
copies Forge's GPT-2 model definition verbatim and adds two marked lines. Your
working `gpt2` program is untouched, so if the conformance build breaks, the
thing you already trust still runs.

The tradeoff: if `gpt2.cpp` changes later, `conformance.cpp` won't follow. The
model code is copied at one point in time. Worth a note in your writeup.

### Step 1 — copy three files

Put `trace_io.hpp`, `trace_tap.hpp`, and `conformance.cpp` into
`experiments/GPT-2/`.

### Step 2 — add two lines to `experiments/CMakeLists.txt`

```cmake
add_executable(conformance GPT-2/conformance.cpp)
target_link_libraries(conformance PRIVATE Forge)
```

(If you're on Windows, also add `conformance` to the `foreach (target gpt2 mnist)`
list further down, so the OpenBLAS DLL gets copied next to it.)

### Step 3 — build

```bash
cmake -B build -DFORGE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target conformance
```

## The two added lines

Inside `TransformerBlock::operator()`, after `auto gelu {m_gelu(l1)};`, a call to
`tap::capture(...)` copies the activation out. Inside `GPT2::operator()`, the
block loop becomes indexed so it can set `tap::current_layer` before each block.

That's the whole instrumentation. It's this small because Forge's GPT-2 runs
**without a batch dimension** — `gelu` is exactly `(seq_len, 3072)`, contiguous,
which is already the layout the trace format expects.

## Determinism: do this before anything else

`main()` in `gpt2.cpp` sets Forge's thread count to your full core count. Forge
parallelises through an **Eigen thread pool**, and `EIGEN_USE_BLAS` hands the
matrix multiplies to **OpenBLAS**, which has its own threading. Both can change
the order floating-point sums are accumulated in, which changes the last few
bits of the result.

So the conformance driver defaults to `--threads 1`, and you should also set:

```bash
export OPENBLAS_NUM_THREADS=1
```

`--threads` does not control OpenBLAS. You need both.

**Prove it before trusting anything.** Run the same case twice and compare the
files byte for byte:

```bash
./build/conformance --tokens fixture/tokens.tsv --out /tmp/run1
./build/conformance --tokens fixture/tokens.tsv --out /tmp/run2
diff -r /tmp/run1 /tmp/run2 && echo "reproducible"
```

If that doesn't come back clean, no comparison against PyTorch means anything
yet, and the problem is threading, not Forge's maths.

## Suggested order

1. **Reproducibility.** The `diff` above, single-threaded. Must pass first.
2. **Weight identity.** Run with `--dump-weight-stats` and compare to
   `weights_fingerprint.json`. The `sum`/`sumsq`/`min`/`max` values must match —
   they're computed in double on both sides and don't care about storage order.
   The `fnv` hash *will* differ for `c_fc` and `c_proj`, because Forge transposes
   those out of Hugging Face's `Conv1D` layout when loading. Matching invariants
   with a differing hash means "same weights, different layout" — expected, and
   worth writing down.
3. **Forge vs PyTorch-CPU.** The actual experiment.
4. **Cost of threading** *(optional)*. Re-run with `--threads 8` and compare
   against your single-threaded run. This measures Forge's own numerical spread,
   the same way cpu-vs-cuda measures PyTorch's.
5. **`-ffast-math`** *(only if needed)*. See below.

## About `-ffast-math`

`CMakeLists.txt` line 103 sets `-mavx2 -ffast-math -fopenmp -mfma` as **PUBLIC**
on the Forge target, so it propagates to anything linking Forge. Turning it off
means editing that line and rebuilding the whole library, not just the driver.

Don't start there. Do steps 1–3 first and see how large the disagreement
actually is. If it's inside the cpu-vs-cuda envelope the comparison script
computes, `-ffast-math` isn't costing you anything worth a rebuild. If it isn't,
then rebuild with `-fno-fast-math -ffp-contract=off` in place of `-ffast-math`
and see how much of the gap closes — that difference is your answer about
whether Forge computes something different or merely rounds differently.

## Pinning the weights — no code change needed

`download_GPT2_124M` skips the download if `./model.safetensors` already exists.
Download the pinned revision yourself and put it in the working directory:

```
https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors
```

The conformance driver never downloads at all — it exits with an error if the
file is missing, rather than silently fetching whatever HEAD is today.

## Two smaller notes

**The tokenizer is out of scope by design.** The driver takes integer ids from
`tokens.tsv` and never calls Forge's BPE. This matters more than it sounds:
Forge's `encode()` applies a hand-rolled `Ġ` substitution to leading spaces, and
loads merge rules from a third repository. Comparing that against Hugging Face's
tokenizer is a worthwhile experiment — just a separate one, with no model in it.
Nothing in Forge prepends token 50256, which is why the fixture carries BOS
explicitly.

**Attention mask constant.** Forge adds `-10000` at future positions where
Hugging Face uses a very large negative number. In float32 both underflow to
zero after the softmax, so I don't expect this to show up. If you ever see a
discrepancy that grows with sequence length, look here.
