# vLLM Gemma 4 Fixes

Official reference repository for resolving Gemma 4 MTP CUDA Graph speculative execution failure and heterogeneous head dimension parameter loading crashes ([vllm-project/vllm Issue #51737](https://github.com/vllm-project/vllm/issues/51737)).

---

## 1. Project Overview & Target Issues

This repository contains critical architectural patches for vLLM to enable stable, high-performance inference for Gemma 4 models (`RedHatAI/gemma-4-31B-it-FP8-dynamic` with `google/gemma-4-31B-it-assistant`).

* **vLLM Issue #51737**: Fixes the parameter loading crash `RuntimeError: start (0) + length (4096) exceeds dimension size (2048)` caused by heterogeneous QKV head dimensions across layers.
* **MTP Optimistic Top-K CUDA Graph Fix**: Fixes speculative decoding freeze and performance degradation where top-k index buffer sharing flags were frozen during CUDA Graph capture.

---

## 2. Root Cause & Technical Mechanics

The issues stem from two primary architectural frictions:

1. **Heterogeneous Head Dimensions**: Gemma 4 employs a mixed-attention architecture (45 sliding-window attention layers with `head_dim = 256`, and 15 full-attention layers with `head_dim = 512`). Standard vLLM weight sharding assumed homogeneous layer parameters, slicing past valid memory bounds during MTP QKV weight loading.
2. **CUDA Graph CPU Flag Freezing**: The MTP speculator attempted to share top-k index buffers across draft steps $1..5$ using CPU-side Python boolean toggling (`set_skip_topk(bool)`). Under CUDA Graph capture, CPU flags are recorded once and frozen. Replaying the captured graph forced the GPU to fall back to re-running expensive top-k logit indexer kernels on every single step.

---

## 3. Code Solutions & Architecture

The fixes implemented here move away from ad-hoc patches toward structural memory safety:

* **Proactive Layer Initialization**: Modified `Gemma4MTPAttention.__init__` to inspect `config.per_layer_config` on layer instantiation so layers configure with their true per-layer `head_dim` (256 vs 512).
* **Defensive Parameter Guard**: Introduced `BasevLLMParameter._safe_narrow()` and vector bounds matching in `default_weight_loader()` to defensively protect against tensor slice overflow.
* **0-Dim GPU Device Tensor Pointer**: Replaced CPU-side flags with a 0-dimensional GPU device memory tensor (`set_skip_topk_tensor`). Captured CUDA Graphs dynamically read the GPU tensor pointer, allowing actual GPU hardware execution to skip top-k logit indexer kernels on steps $1..5$ without triggering CPU-GPU syncs or graph recompilations.

---

## 4. Empirical Infrastructure Benchmarks

Live benchmarks conducted on our NVIDIA H200 GPU infrastructure (`RedHatAI/gemma-4-31B-it-FP8-dynamic` + `google/gemma-4-31B-it-assistant`):

| Metric | Result | Infrastructure Notes |
| :--- | :--- | :--- |
| **Short Context Throughput** | **190.65 tok/s** | 24 prompt tokens, 256 max completion tokens |
| **Long Context Throughput** | **97.39 tok/s** | 2,543 prompt tokens (full agentic context) |
| **Throughput Improvement** | **+22.8% to +42.3%** | Vs. unpatched vLLM re-sorting top-k on every step |
| **Draft Acceptance Rate** | **51.3% Avg Acceptance** | Per-position: `[0.771, 0.617, 0.479, 0.383, 0.314]` |
| **Mean Speculative Length** | **3.56 tokens** | Per speculative step |

---

## 5. Quick Start & How to Apply

### Prerequisites
* vLLM `0.27.0-dev` or `main`
* PyTorch 2.4+ & CUDA 12.x

### Applying to vLLM
To apply the patches to a local `vllm` repository:

```bash
cd vllm
# Switch to MTP CUDA Graph-safe Top-K fix branch
git checkout mtp-cuda-graph-topk-fix
```

Or apply the patch file directly:
```bash
git apply 0001-mtp-speculator-cuda-graph-safe-topk-sharing.patch
```
