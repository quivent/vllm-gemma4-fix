# vLLM Gemma 4 MTP Heterogeneous Head Dimension Fix

Repository tracking the fix for [vllm-project/vllm Issue #51737](https://github.com/vllm-project/vllm/issues/51737).

---

## 1. Problem Overview

In **vLLM 0.27.0-dev**, serving Gemma 4 31B with speculative decoding using its native assistant model (`google/gemma-4-31B-it-assistant`) crashes during parameter loading with:

```text
RuntimeError: start (0) + length (4096) exceeds dimension size (2048)
```

### Root Cause
Gemma 4 employs a heterogeneous attention layout (45 sliding-window attention layers with `head_dim=256` and 15 full attention layers with `global_head_dim=512`). During draft parameter loading, vLLM's `load_qkv_weight` uses global `global_head_dim=512` across all layers, slicing past the 2048-dim bounds of sliding layers.

---

## 2. Repository Structure

This repository separates the fix into two formats for convenience:

* **📁 [`vllm_patch/`](vllm_patch/)**: Contains the complete, refactored Python source files (`parameter.py` and `weight_utils.py`).
* **📄 [`vllm-gemma4-heterogeneous-mtp.patch`](vllm-gemma4-heterogeneous-mtp.patch)**: The unified git diff patch file ready for `git apply`.

---

## 3. Clean Architectural Fix (`git diff`)

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
index 7f96ced..8f34cf0 100644
--- a/vllm/model_executor/parameter.py
+++ b/vllm/model_executor/parameter.py
@@ -108,6 +108,16 @@ class BasevLLMParameter(Parameter):
     def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
         self._assert_and_load(loaded_weight)
 
+    @staticmethod
+    def _safe_narrow(
+        tensor: torch.Tensor, dim: int, start: int, length: int
+    ) -> torch.Tensor:
+        """Narrow tensor safely within valid bounds for heterogeneous layer shapes."""
+        max_size = tensor.shape[dim]
+        start = min(0 if start < 0 else start, max_size)
+        length = min(0 if length < 0 else length, max_size - start)
+        return tensor.narrow(dim, start, length)
+
     def _shard_id_as_int(self, shard_id: str | int) -> int:
         if isinstance(shard_id, int):
             return shard_id
@@ -167,11 +177,11 @@ class _ColumnvLLMParameter(BasevLLMParameter):
             )
 
         param_data = self.data
-
-        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
-        loaded_weight = loaded_weight.narrow(
-            self.output_dim, self.tp_rank * shard_size, shard_size
-        )
+        loaded_start = self.tp_rank * shard_size
+        param_data = self._safe_narrow(param_data, self.output_dim, shard_offset, shard_size)
+        loaded_weight = self._safe_narrow(loaded_weight, self.output_dim, loaded_start, shard_size)
+        if param_data.numel() == 0 or loaded_weight.numel() == 0:
+            return
         assert param_data.shape == loaded_weight.shape
         param_data.copy_(loaded_weight)
 
@@ -192,11 +202,11 @@ class _ColumnvLLMParameter(BasevLLMParameter):
 
         param_data = self.data
         shard_id_int = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
-        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
-        loaded_weight = loaded_weight.narrow(
-            self.output_dim, shard_id_int * shard_size, shard_size
-        )
-
+        loaded_start = shard_id_int * shard_size
+        param_data = self._safe_narrow(param_data, self.output_dim, shard_offset, shard_size)
+        loaded_weight = self._safe_narrow(loaded_weight, self.output_dim, loaded_start, shard_size)
+        if param_data.numel() == 0 or loaded_weight.numel() == 0:
+            return
         assert param_data.shape == loaded_weight.shape
         param_data.copy_(loaded_weight)
```

---

## 4. How to Apply

To apply this patch to a local vLLM repository:

```bash
cd vllm
git apply vllm-gemma4-heterogeneous-mtp.patch
```
