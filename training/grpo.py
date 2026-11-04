"""GRPO integration boundary. Install the trainer and supply its rollout runner."""
import argparse

from common.config import check_output_path, load_config, load_extension, required
from training.reward import RewardConfig, score_group, validate_group
from training.reward_interface import judge_group


class GroupRewardHook:
    """Call after collecting all same-input, same-policy rollouts in a group."""
    def __init__(self, config):
        self.config = RewardConfig(**config)

    def __call__(self, rows, reference, *, verdicts=None, judge=None):
        validate_group(rows, reference, self.config)
        if (verdicts is None) == (judge is None):
            raise ValueError("Supply either verdicts or a judge, but not both")
        if verdicts is None:
            verdicts = judge_group(rows, reference, judge)
        return score_group(rows, reference, verdicts, self.config)


def terminal_rewards(response_mask, scored_rows, trajectory_ids):
    """Place scalar scores on the final generated token, never on tool tokens.

    A veRL adapter can use this as token_level_scores before the framework's
    existing KL handling and GRPO advantage computation. No advantage is computed here.
    """
    import torch

    if response_mask.ndim != 2 or not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be a binary [batch, response] tensor")
    if len(trajectory_ids) != response_mask.shape[0] or len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("Provide unique IDs in exact rollout tensor order")
    lookup = {row["trajectory_id"]: row["reward"] for row in scored_rows}
    if len(lookup) != len(scored_rows) or set(lookup) != set(trajectory_ids):
        raise ValueError("Reward IDs and rollout IDs must match exactly")
    result = torch.zeros_like(response_mask, dtype=torch.float32)
    for index, ident in enumerate(trajectory_ids):
        positions = response_mask[index].bool().nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("Cannot attach reward to a response without generated tokens")
        value = float(lookup[ident])
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("Nonfinite scalar reward")
        result[index, positions[-1]] = value
    return result


def run(config):
    if config.get("algorithm") != "grpo":
        raise ValueError("This entry is for GRPO only")
    hook = GroupRewardHook(required(config, "reward"))
    runner = load_extension(required(config, "runner"))
    model_path = required(config, "model_path")
    check_output_path(required(config, "output_dir"), model_path)
    # The external runner owns model loading, generation, tool/media execution,
    # the framework's optimizer/advantage math, distribution and checkpoints.
    # Supplying a module explicitly is required; there is no silent fallback.
    return runner(config=config, group_reward=hook, terminal_rewards=terminal_rewards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--runner", help="External trainer adapter as module:callable")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("runner", "model_path", "output_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run(config)


if __name__ == "__main__":
    main()
