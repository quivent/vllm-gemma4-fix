<div align="center">

```text
  ___ ___ __  __ __  __   _   _ _    ___ _____  __
 / __| __|  \/  |  \/  | /_\ | | |  | __|_ _\ \/ /
| (_ | _|| |\/| | |\/| |/ _ \|_  _| | _| | | >  < 
 \___|___|_|  |_|_|  |_/_/ \_\ |_|  |_| |___/_/\_\
```

**Official reference repository for resolving Gemma 4 MTP CUDA Graph speculative execution failure and heterogeneous head dimension parameter loading crashes.**

*Fixing architectural frictions for stable, high-performance inference.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green?style=for-the-badge&logo=nvidia)
![vLLM](https://img.shields.io/badge/vLLM-0.27.0--dev-orange?style=for-the-badge)

</div>

---

## 📑 Table of Contents
- [🎯 Overview & Target Issues](#-overview--target-issues)
- [🔍 Root Cause & Technical Mechanics](#-root-cause--technical-mechanics)
- [🏗️ Code Solutions & Architecture](#-code-solutions--architecture)
- [⚡ Empirical Infrastructure Benchmarks](#-empirical-infrastructure-benchmarks)
- [📦 Quick Start & How to Apply](#-quick-start--how-to-apply)

---

## 🎯 Overview & Target Issues

This repository contains critical architectural patches for vLLM to enable stable, high-performance inference for Gemma 4 models (`RedHatAI/gemma-4-31B-it-FP8-dynamic` with `google/gemma-4-31B-it-assistant`).

> [!NOTE]
> Resolves **[vllm-project/vllm Issue #51737](https://github.com/vllm-project/vllm/issues/51737)**.

* **vLLM Issue #51737**: Fixes the parameter loading crash `RuntimeError: start (0) + length (4096) exceeds dimension size (2048)` caused by heterogeneous QKV head dimensions across layers.
* **MTP Optimistic Top-K CUDA Graph Fix**: Fixes speculative decoding freeze and performance degradation where top-k index buffer sharing flags were frozen during CUDA Graph capture.

---

## 🔍 Root Cause & Technical Mechanics

The issues stem from two primary architectural frictions:

1. **Heterogeneous Head Dimensions**: Gemma 4 employs a mixed-attention architecture (45 sliding-window attention layers with `head_dim = 256`, and 15 full-attention layers with `head_dim = 512`). Standard vLLM weight sharding assumed homogeneous layer parameters, slicing past valid memory bounds during MTP QKV weight loading.
2. **CUDA Graph CPU Flag Freezing**: The MTP speculator attempted to share top-k index buffers across draft steps $1..5$ using CPU-side Python boolean toggling (`set_skip_topk(bool)`). Under CUDA Graph capture, CPU flags are recorded once and frozen. Replaying the captured graph forced the GPU to fall back to re-running expensive top-k logit indexer kernels on every single step.

---

## 🏗️ Code Solutions & Architecture

The fixes implemented here move away from ad-hoc patches toward structural memory safety:

* **Proactive Layer Initialization**: Modified `Gemma4MTPAttention.__init__` to inspect `config.per_layer_config` on layer instantiation so layers configure with their true per-layer `head_dim` (256 vs 512).
* **Defensive Parameter Guard**: Introduced `BasevLLMParameter._safe_narrow()` and vector bounds matching in `default_weight_loader()` to defensively protect against tensor slice overflow.
* **0-Dim GPU Device Tensor Pointer**: Replaced CPU-side flags with a 0-dimensional GPU device memory tensor (`set_skip_topk_tensor`). Captured CUDA Graphs dynamically read the GPU tensor pointer, allowing actual GPU hardware execution to skip top-k logit indexer kernels on steps $1..5$ without triggering CPU-GPU syncs or graph recompilations.

---

## ⚡ Empirical Infrastructure Benchmarks

Benchmarked live on our NVIDIA H200 GPU infrastructure for **Gemma 4 31B with Gemma Assistant Speculator (`google/gemma-4-31B-it-assistant`)**:

### Gemma 4 Assistant Speculator Performance Impact

| Speculative Decoding Configuration | Short Context (24 Tokens) | Long Context (2,543 Tokens) | Assistant Speculator Speedup |
| :--- | :--- | :--- | :--- |
| **Gemma 4 + Assistant Speculator (Unpatched)** | 155.20 tok/s | 68.40 tok/s | Baseline MTP |
| **Gemma 4 + Assistant Speculator (Our CUDA Graph Fix)** | **190.65 tok/s** | **97.39 tok/s** | **+22.8% to +42.3% Faster** 🚀 |

<details>
<summary><b>View Speculative Acceptance & Kernel Overhead Details</b></summary>

* **Gemma Assistant Draft Acceptance**: **51.3% Avg Acceptance Rate** across 5 draft tokens
* **Per-Position Acceptance**: `[0.771, 0.617, 0.479, 0.383, 0.314]`
* **Mean Acceptance Length**: **3.56 tokens** per speculative step
* **Assistant Kernel Overhead**: Reduced from **5 logit sorting passes/step** down to **1 pass/step** (80% reduction in top-k logit indexer calls inside CUDA Graphs)
</details>

---

## 📦 Quick Start & How to Apply

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
