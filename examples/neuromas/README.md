# NeuroMAS

[NeuroMAS](https://arxiv.org/abs/2605.16757) treats a multi-agent language system as a trainable,
neural-network-like architecture: language-agent nodes are connected by text-carrying edges, no node is
hand-assigned a role, and the whole network is optimized jointly from the final task reward alone.

This example runs that training loop on verl. It drives the standard `verl.trainer.main_ppo` entry point through
a custom [agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), with no change to the trainer,
the loss, or the reward functions.

Reference: [NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning](https://arxiv.org/abs/2605.16757),
*The First Conference on Statistics and Trustworthy AI for Cross (X)-Domain Acceleration (STAI-X 2026)*, Paper Award Honorable Mention.

## Canonical Scripts

| Script                    | Infer | Train | Platform | Dataset |
|---------------------------|-------|-------|----------|---------|
| `run_qwen3_0.6b_fsdp.sh`  | vLLM  | FSDP  | NVIDIA   | GSM8K   |

## How it works

`NeuroMASAgentLoop` in [`neuromas_agent_loop.py`](neuromas_agent_loop.py) runs each query through a chain of nodes
and an output aggregation node. The default, `num_layers: 2`, is NeuroMAS-3: three LLM calls per query.

1. **Structure-aware, role-free nodes.** Each node's prompt states only its position in the network and the
   message format, for example `[Network: 2-layer 1-1, 2 nodes | Layer 1/2, Position 1/1]`. No role such as
   planner or verifier is assigned by hand.
2. **Text-passing forward pass.** Each node writes a message for the next node. The aggregator reads the node
   outputs and produces the answer.
3. **One shared terminal reward.** Only the aggregator's answer is scored, using verl's built-in reward for the
   dataset. All nodes are serialized into one trajectory with `response_mask=1` on generated tokens and `0` on the
   routing prompts, so every node's tokens are trained from that single reward.
4. **Joint policy-gradient update.** The script uses `loss_mode=gpg`, a plain log-probability policy gradient, with
   the `reinforce_plus_plus_baseline` advantage (each sample's reward minus the mean over its `rollout.n` samples).

Nodes share one policy in this example and are distinguished by their prompts. Per-node LoRA adapters, wider
topologies, and progressive growth (NeuroMAS-3 to -5 to -7) are in the
[reference implementation](https://github.com/haoranlustat/NeuroMAS).

## Quick start

From the verl repo root:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k
bash examples/neuromas/run_qwen3_0.6b_fsdp.sh
```

The script trains on GSM8K. To use another dataset with a verl built-in reward, point `data.train_files` and
`data.val_files` at its parquet files.

## Configuration

Topology and generation budget live in [`agent.yaml`](agent.yaml):

| Key               | Default | Meaning                                                               |
|-------------------|---------|-----------------------------------------------------------------------|
| `num_layers`      | `2`     | Node layers. LLM calls per query = `num_layers + 1`.                  |
| `node_max_tokens` | `256`   | Generation budget of each LLM call.                                   |

Keep `MAX_RESPONSE_LENGTH` above `(num_layers + 1) * node_max_tokens` plus room for the routing prompts, since the
whole chain is one trajectory.

## Citation

```bibtex
@inproceedings{lu2026neuromas,
  title     = {NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning},
  author    = {Lu, Haoran and Fang, Luyang and Zhong, Wenxuan and Ma, Ping},
  booktitle = {The First Conference on Statistics and Trustworthy AI for Cross (X)-Domain Acceleration (STAI-X 2026)},
  year      = {2026}
}
```
