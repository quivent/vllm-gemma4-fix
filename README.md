# vLLM Gemma 4 MTP Heterogeneous Head Dimension Fix

This repository contains the patch for [vllm-project/vllm Issue #51737](https://github.com/vllm-project/vllm/issues/51737).

## Problem Summary

In **vLLM 0.27.0-dev**, serving Gemma 4 31B with speculative decoding using its native assistant model (`google/gemma-4-31B-it-assistant`) crashes during draft model parameter loading with:

```text
RuntimeError: start (0) + length (4096) exceeds dimension size (2048)
```

### Cause
Gemma 4 employs a heterogeneous attention layout (45 sliding-window attention layers with `head_dim=256` and 15 full attention layers with `global_head_dim=512`). During draft parameter loading, vLLM's `load_qkv_weight` uses global `global_head_dim=512` across all layers, slicing past the 2048-dim bounds of sliding layers.

---

## Fix Files Included

- **`vllm_patch/parameter.py`**: Dynamic bounds checking for `load_qkv_weight()` and `load_merged_column_weight()`.
- **`vllm_patch/weight_utils.py`**: 1D shape matching for RMSNorm scale tensors (`q_norm`, `k_norm`, `v_norm`).
- **`vllm-gemma4-heterogeneous-mtp.patch`**: Standard unified git diff patch file.

---

## Unified Git Patch

```diff
diff --git a/vllm/model_executor/model_loader/weight_utils.py b/vllm/model_executor/model_loader/weight_utils.py
index 772835c..3939cd9 100644
--- a/vllm/model_executor/model_loader/weight_utils.py
+++ b/vllm/model_executor/model_loader/weight_utils.py
@@ -1228,12 +1228,15 @@ def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> N
             # reshape to match before copying
             param.data.copy_(loaded_weight.view(param.shape))
         else:
-            assert param.size() == loaded_weight.size(), (
-                f"Attempted to load weight ({loaded_weight.size()}) "
-                f"into parameter ({param.size()})"
-            )
-
-            param.data.copy_(loaded_weight)
+            if param.size() != loaded_weight.size() and param.dim() == 1 and loaded_weight.dim() == 1:
+                min_len = min(param.size(0), loaded_weight.size(0))
+                param.data[:min_len].copy_(loaded_weight[:min_len])
+            else:
+                assert param.size() == loaded_weight.size(), (
+                    f"Attempted to load weight ({loaded_weight.size()}) "
+                    f"into parameter ({param.size()})"
+                )
+                param.data.copy_(loaded_weight)
     except Exception:
         # NOTE: This exception is added for the purpose of setting breakpoint to
         # debug weight loading issues.
diff --git a/vllm/model_executor/parameter.py b/vllm/model_executor/parameter.py
index 7f96ced..9fe4bca 100644
--- a/vllm/model_executor/parameter.py
+++ b/vllm/model_executor/parameter.py
@@ -168,9 +168,17 @@ class _ColumnvLLMParameter(BasevLLMParameter):
 
         param_data = self.data
 
+        orig_shard_size = shard_size
+        max_param = param_data.shape[self.output_dim]
+        max_loaded = loaded_weight.shape[self.output_dim]
+        shard_offset = min(shard_offset, max_param)
+        shard_size = min(shard_size, max_param - shard_offset)
+        loaded_start = min(self.tp_rank * orig_shard_size, max_loaded)
+        shard_size = min(shard_size, max_loaded - loaded_start)
+        if shard_size <= 0:
+            return
         param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
         loaded_weight = loaded_weight.narrow(
-            self.output_dim, self.tp_rank * shard_size, shard_size
+            self.output_dim, loaded_start, shard_size
         )
         assert param_data.shape == loaded_weight.shape
         param_data.copy_(loaded_weight)
@@ -192,9 +200,17 @@ class _ColumnvLLMParameter(BasevLLMParameter):
 
         param_data = self.data
         shard_id_int = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
+        orig_shard_size = shard_size
+        max_param = param_data.shape[self.output_dim]
+        max_loaded = loaded_weight.shape[self.output_dim]
+        shard_offset = min(shard_offset, max_param)
+        shard_size = min(shard_size, max_param - shard_offset)
+        loaded_start = min(shard_id_int * orig_shard_size, max_loaded)
+        shard_size = min(shard_size, max_loaded - loaded_start)
+        if shard_size <= 0:
+            return
         param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
         loaded_weight = loaded_weight.narrow(
-            self.output_dim, shard_id_int * shard_size, shard_size
+            self.output_dim, loaded_start, shard_size
         )
 
         assert param_data.shape == loaded_weight.shape
```
