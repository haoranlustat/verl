# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Individual Contributor: Haoran Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput

_MODULE_PATH = Path(__file__).parents[3] / "examples" / "neuromas" / "neuromas_agent_loop.py"
_spec = importlib.util.spec_from_file_location("neuromas_agent_loop", _MODULE_PATH)
neuromas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(neuromas)

_GENERATION_PROMPT = 999
_ROLE_TOKENS = {"system": 501, "user": 502, "assistant": 503}


class _FakeTokenizer:
    """Render each message as a role token plus one token per content word."""

    padding_side = "right"
    pad_token_id = 0

    def __init__(self):
        self.texts: dict[tuple[int, ...], str] = {}

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=True, tokenize=True, **kwargs):
        del tools, tokenize, kwargs
        token_ids = []
        for message in messages:
            token_ids.append(_ROLE_TOKENS[message["role"]])
            token_ids.extend(1000 + len(word) for word in message["content"].split())
        if add_generation_prompt:
            token_ids.append(_GENERATION_PROMPT)
        return token_ids

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return self.texts[tuple(ids)]


class _FakeServerManager:
    """Return one scripted reply per LLM call and record each request."""

    def __init__(self, tokenizer: _FakeTokenizer, replies: list[str]):
        self.tokenizer = tokenizer
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.generated: list[list[int]] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs) -> TokenOutput:
        del request_id, kwargs
        self.requests.append({"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        call = len(self.generated)
        token_ids = [2000 + call * 10 + i for i in range(3)]
        self.tokenizer.texts[tuple(token_ids)] = self.replies[call]
        self.generated.append(token_ids)
        return TokenOutput(token_ids=token_ids, log_probs=[-0.5] * len(token_ids))


def _make_loop(replies: list[str], response_length: int = 512, num_layers: int = 2):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 512, "response_length": response_length, "full_determinism": False},
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager(tokenizer, replies)
    loop = neuromas.NeuroMASAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        num_layers=num_layers,
        node_max_tokens=64,
    )
    return loop, server_manager


def _sample() -> dict[str, Any]:
    return {
        "raw_prompt": [{"role": "user", "content": 'What is 40 + 2? Output the final answer after "####".'}],
        "data_source": "openai/gsm8k",
        "reward_model": {"style": "rule", "ground_truth": "42"},
        "extra_info": {},
    }


def test_node_prompts_are_structure_aware_and_role_free():
    first = neuromas.node_prompt(2, 0, "Q?")
    last = neuromas.node_prompt(2, 1)

    assert "[Network: 2-layer 1-1, 2 nodes | Layer 1/2, Position 1/1]" in first
    assert "PROBLEM:\nQ?" in first
    assert "TO #1:" in first
    assert "Layer 2/2" in last
    assert "TO #1:" not in last
    assert "PROBLEM" not in last
    for prompt in (first, last, neuromas.aggregator_prompt(2)):
        for role in ("planner", "critic", "verifier"):
            assert role not in prompt.lower()


@pytest.mark.asyncio
async def test_all_nodes_form_one_trajectory_with_generated_tokens_trainable():
    loop, server_manager = _make_loop(["TO #1: add them", "It is 42", "#### 42"])

    output = await loop.run(sampling_params={"temperature": 1.0}, **_sample())

    assert len(server_manager.requests) == 3
    assert all(request["sampling_params"]["max_tokens"] == 64 for request in server_manager.requests)

    generated = {token for token_ids in server_manager.generated for token in token_ids}
    assert sum(output.response_mask) == 9
    for token, mask in zip(output.response_ids, output.response_mask, strict=True):
        assert mask == int(token in generated)

    # Each later node sees everything generated before it.
    assert server_manager.requests[1]["prompt_ids"][: len(output.prompt_ids) + 3] == (
        output.prompt_ids + server_manager.generated[0]
    )
    assert output.response_logprobs is not None
    assert len(output.response_logprobs) == len(output.response_ids)
    assert output.num_turns == 6


@pytest.mark.asyncio
async def test_terminal_reward_scores_the_aggregator_answer():
    loop, _ = _make_loop(["TO #1: add them", "It is 41", "#### 42"])

    output = await loop.run(sampling_params={}, **_sample())

    assert output.reward_score == 1.0
    assert output.extra_fields["reward_extra_info"] == {"acc": 1.0}


@pytest.mark.asyncio
async def test_upstream_answers_do_not_earn_the_reward():
    loop, _ = _make_loop(["#### 42", "#### 42", "#### 7"])

    output = await loop.run(sampling_params={}, **_sample())

    assert output.reward_score == 0.0


@pytest.mark.asyncio
async def test_trajectory_ending_before_the_aggregator_scores_zero():
    # The first prompt is in prompt_ids; 3 generated tokens plus the next prompt exhaust this budget.
    loop, server_manager = _make_loop(["#### 42", "#### 42", "#### 42"], response_length=4)

    output = await loop.run(sampling_params={}, **_sample())

    assert len(server_manager.requests) == 1
    assert output.reward_score == 0.0
    assert len(output.response_ids) == 4


def test_num_layers_must_be_positive():
    with pytest.raises(ValueError, match="num_layers"):
        _make_loop([], num_layers=0)
