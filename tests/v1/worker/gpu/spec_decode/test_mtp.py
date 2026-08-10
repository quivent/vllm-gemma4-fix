# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock
import torch
import pytest

from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


class MockModel:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_skip_topk = MagicMock()
        self.set_skip_topk_tensor = MagicMock()
        self.compact_topk_indices = MagicMock()


class MockConfig:

    def __init__(self):
        self.speculative_config = MagicMock()
        self.speculative_config.draft_model_config.hf_config = MagicMock()
        self.speculative_config.draft_model_config.hf_config.index_share_for_mtp_iteration = True


def test_mtp_speculator_state_transitions():
    config = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speculator = MTPSpeculator(config, device)
    speculator.num_speculative_steps = 2

    mock_model = MockModel()
    speculator.model = MagicMock()
    speculator.model.model = mock_model
    speculator.last_token_indices = torch.randn(10)
    speculator.share_mtp_topk_indices = True

    # 1. Test Prefill Begin (resets flag on tensor)
    speculator.on_prefill_begin(num_reqs=5)
    assert mock_model.set_skip_topk_tensor.called or mock_model.set_skip_topk.called

    # 2. Test Prefill End (compacts top-k index buffer)
    speculator.on_prefill_end(num_reqs=5)
    mock_model.compact_topk_indices.assert_called_once()

    # 3. Test Decode Begin (toggles reuse mode on GPU tensor)
    speculator.on_multi_step_decode_begin(num_reqs=5)
    assert mock_model.set_skip_topk_tensor.called or mock_model.set_skip_topk.called

    # 4. Test Decode End (resets reuse mode)
    speculator.on_multi_step_decode_end(num_reqs=5)
    assert mock_model.set_skip_topk_tensor.called or mock_model.set_skip_topk.called


def test_mtp_speculator_disabled_transitions():
    config = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speculator = MTPSpeculator(config, device)
    speculator.share_mtp_topk_indices = False
    mock_model = MockModel()
    speculator.model = MagicMock()
    speculator.model.model = mock_model

    speculator.on_prefill_begin(num_reqs=5)
    mock_model.set_skip_topk_tensor.assert_not_called()
    mock_model.set_skip_topk.assert_not_called()
