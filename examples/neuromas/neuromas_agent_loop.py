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
"""NeuroMAS-style multi-agent rollout on the verl agent loop.

A query runs through a chain of role-free language-agent nodes followed by an
output aggregation node. Each node knows only its position in the network, and
writes a message for the next node. Every generated token is trained from one
shared terminal reward: the score of the aggregator's answer.

Reference: https://arxiv.org/abs/2605.16757
"""

from typing import Any, ClassVar
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.reward_score import default_compute_score
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


def network_description(num_layers: int) -> str:
    """Describe the chain topology, e.g. ``2-layer 1-1, 2 nodes``."""
    widths = "-".join(["1"] * num_layers)
    return f"{num_layers}-layer {widths}, {num_layers} nodes"


def node_prompt(num_layers: int, layer: int, question: str | None = None) -> str:
    """Build the structure-aware, role-free prompt of the node at ``layer``.

    The first node receives ``question``. Later nodes read the conversation so far,
    which already holds the problem and the upstream message.
    """
    header = f"[Network: {network_description(num_layers)} | Layer {layer + 1}/{num_layers}, Position 1/1]"
    parts = [header]
    if question is not None:
        parts.append(f"PROBLEM:\n{question}")
    if layer > 0:
        parts.append("The message from the previous node is above.")
    if layer < num_layers - 1:
        parts.append("Write a private message to the next node:\nTO #1: <message for downstream node 1>")
    else:
        parts.append("Answer the problem.")
    return "\n\n".join(parts)


def aggregator_prompt(num_layers: int) -> str:
    """Build the prompt of the output aggregation node."""
    return (
        f"[Summarizer | Network: {network_description(num_layers)}]\n\n"
        "All node outputs are above. Give the final answer in the format the problem asks for."
    )


class NeuroMASAgentLoop(AgentLoopBase):
    """Run one query through a chain of nodes and an aggregator, scored at the aggregator.

    The nodes are serialized into a single trajectory: generated tokens have
    ``response_mask=1`` and the routing prompts have ``response_mask=0``, so every node's
    tokens receive the terminal reward from the standard outcome-reward advantage estimators.

    Args:
        num_layers: Number of node layers. The system makes ``num_layers + 1`` LLM calls, so
            the default of 2 is the paper's NeuroMAS-3.
        node_max_tokens: Generation budget of each LLM call.
    """

    merges_context_messages: ClassVar[bool] = True

    def __init__(self, *args, num_layers: int = 2, node_max_tokens: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        self.num_layers = num_layers
        self.node_max_tokens = node_max_tokens
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        request_id = f"det-{int(priority)}" if self.rollout_config.get("full_determinism", False) else uuid4().hex
        messages = list(kwargs["raw_prompt"])
        question = messages[-1]["content"]
        messages[-1] = {**messages[-1], "content": node_prompt(self.num_layers, 0, question)}
        next_prompts = [node_prompt(self.num_layers, layer) for layer in range(1, self.num_layers)]
        next_prompts.append(aggregator_prompt(self.num_layers))

        metrics: dict[str, Any] = {}
        extra_fields: dict[str, Any] = {}
        token_ids = await self.ct_build_initial_tokens(messages)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        aggregator_text = None

        for call in range(self.num_layers + 1):
            if call > 0:
                previous_messages = list(messages)
                messages.append({"role": "user", "content": next_prompts[call - 1]})
                merge_result, response_mask, merged_logprobs = await self.ct_merge_context_msg(
                    previous_messages, messages, token_ids, response_mask, response_logprobs or None
                )
                token_ids = merge_result.token_ids
                response_logprobs = merged_logprobs or []

            budget = min(self.node_max_tokens, self.response_length - len(response_mask))
            if budget <= 0:
                break

            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=token_ids,
                    sampling_params={**sampling_params, "max_tokens": budget},
                )
            metrics["num_preempted"] = metrics.get("num_preempted", 0) + (output.num_preempted or 0)
            extra_fields.update(output.extra_fields)

            merge_result, response_mask, merged_logprobs = await self.ct_merge_assistant_token(
                token_ids,
                output.token_ids,
                response_mask,
                response_logprobs if (response_logprobs or output.log_probs) else None,
                assistant_logprobs=output.log_probs or None,
            )
            token_ids = merge_result.token_ids
            response_logprobs = merged_logprobs or []

            text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            if call == self.num_layers:
                aggregator_text = text

        score = await self._terminal_score(aggregator_text, kwargs)
        extra_fields.update({"turn_scores": [], "tool_rewards": [], "reward_extra_info": {"acc": score}})

        response_ids = token_ids[-len(response_mask) :]
        return AgentLoopOutput(
            prompt_ids=token_ids[: len(token_ids) - len(response_mask)],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] or None,
            reward_score=score,
            num_turns=len(messages),
            metrics=metrics,
            extra_fields=extra_fields,
        )

    async def _terminal_score(self, aggregator_text: str | None, sample: dict[str, Any]) -> float:
        """Score only the aggregator's answer, so upstream messages cannot earn the reward."""
        if aggregator_text is None:
            return 0.0
        result = await self.loop.run_in_executor(
            None,
            lambda: default_compute_score(
                data_source=sample["data_source"],
                solution_str=aggregator_text,
                ground_truth=sample["reward_model"]["ground_truth"],
                extra_info=sample.get("extra_info"),
            ),
        )
        return float(result["score"] if isinstance(result, dict) else result)
