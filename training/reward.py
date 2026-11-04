"""Local reward arithmetic; semantic evidence is supplied by an external judge."""
from dataclasses import dataclass
import hashlib
import json
import math


class JudgeError(RuntimeError):
    """Missing/uncertain evidence is a technical failure, not a zero reward."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def reference_digest(reference):
    serialized = json.dumps(reference, sort_keys=True, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def atoms(items, *, allow_empty=False):
    require(isinstance(items, list) and (items or allow_empty), "Expected an atom list")
    require(all(isinstance(item, dict) and nonempty(item.get("id")) and nonempty(item.get("text"))
                for item in items), "Atoms require id and text")
    require(len({item["id"] for item in items}) == len(items), "Duplicate atom ID")
    require(len({" ".join(item["text"].lower().split()) for item in items}) == len(items),
            "Duplicate atom text; semantic deduplication is the judge's responsibility")
    require(not any("weight" in item for item in items), "Per-atom weights are not supported")
    return {item["id"] for item in items}


@dataclass(frozen=True)
class RewardConfig:
    quality_threshold: float
    observation_budget: int  # Maximum tool calls after the shared global preview.
    group_size: int = 8
    alpha: float = 0.8
    beta: float = 0.2
    mu: float = 0.05
    failure_reward: float = -1.0

    def __post_init__(self):
        require(type(self.quality_threshold) in (float, int) and 0 < self.quality_threshold <= 1,
                "quality_threshold must be in (0, 1]")
        require(type(self.observation_budget) is int and self.observation_budget > 0,
                "Supply a positive tool-call budget after the shared global preview")
        require(type(self.group_size) is int and self.group_size >= 2, "Invalid group size")
        require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for value in (self.alpha, self.beta, self.mu)), "Invalid reward coefficients")
        require(math.isfinite(self.failure_reward) and self.failure_reward <= -self.mu,
                "Invalid policy outputs must not outrank valid low-quality outputs")


def matching_score(section, reference_atoms, source):
    """One-to-one maximum matching prevents duplicate edges inflating F1."""
    require(isinstance(section, dict) and set(section) == {"generated", "matches"},
            "Expected generated atoms and matching edges")
    left = atoms(section["generated"], allow_empty=True)
    right = atoms(reference_atoms)
    for item in section["generated"]:
        quote = item.get("source_quote")
        require(nonempty(quote) and quote in source, "An atom lacks a verbatim output span")
    require(isinstance(section["matches"], list), "Expected a match list")
    adjacency = {key: set() for key in left}
    for edge in section["matches"]:
        require(isinstance(edge, dict) and edge.get("generated_id") in left
                and edge.get("reference_id") in right and nonempty(edge.get("reason")),
                "Match cites an unknown atom or lacks a reason")
        adjacency[edge["generated_id"]].add(edge["reference_id"])
    matched = {}

    def augment(node, visited):
        for target in sorted(adjacency[node]):
            if target in visited:
                continue
            visited.add(target)
            if target not in matched or augment(matched[target], visited):
                matched[target] = node
                return True
        return False

    for node in sorted(left):
        augment(node, set())
    count = len(matched)
    precision = count / len(left) if left else 0.0
    recall = count / len(right)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, matched=count,
                generated=len(left), reference=len(right), pairs=sorted((v, k) for k, v in matched.items()))


def validate_group(rows, reference, config):
    require(isinstance(reference, dict) and nonempty(reference.get("id")), "Reference ID required")
    atoms(reference["facts"])
    atoms(reference["plan_points"])
    require(len(rows) == config.group_size, "Score complete groups only")
    require(all(nonempty(row.get("trajectory_id")) for row in rows), "Trajectory ID required")
    require(len({row["trajectory_id"] for row in rows}) == len(rows), "Duplicate trajectory")
    for key in ("group_id", "policy_version", "observation_digest"):
        require(all(nonempty(row.get(key)) for row in rows)
                and len({row[key] for row in rows}) == 1, "Group mixes " + key)
    fingerprint = reference_digest(reference)
    for row in rows:
        require(row["reference_id"] == reference["id"] and row["reference_digest"] == fingerprint,
                "Reference differs within the group or changed after generation")
        if row.get("technical_failure"):
            raise JudgeError("Do not turn an infrastructure failure into a policy penalty")
        require(type(row.get("valid")) is bool, "Explicit policy validity required")
        cost = row.get("extra_tool_calls")
        require(type(cost) is int and cost >= 0,
                "extra_tool_calls must count tool invocations after the shared global preview")
        if row["valid"]:
            require(cost <= config.observation_budget, "Valid rollout exceeded the tool-call budget")
            require(nonempty(row.get("final_caption")), "Final caption required")
            require(isinstance(row.get("initial_plan"), str), "Initial plan required")
        else:
            require(nonempty(row.get("failure_reason")), "Policy failure reason required")


def score_group(rows, reference, verdicts, config):
    """Return ordered scalar rewards; the external trainer owns GRPO advantages."""
    validate_group(rows, reference, config)
    scores = {}
    for row in rows:
        if not row["valid"]:
            continue
        ident = row["trajectory_id"]
        try:
            verdict = verdicts[ident]
            require(isinstance(verdict, dict) and verdict.get("status") == "ok",
                    "The judge must return complete, certain evidence")
            scores[ident] = dict(
                kfr=matching_score(verdict["caption"], reference["facts"], row["final_caption"]),
                par=matching_score(verdict["plan"], reference["plan_points"], row["initial_plan"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise JudgeError(str(exc)) from exc
    gate = any(item["kfr"]["f1"] >= config.quality_threshold for item in scores.values())
    results = []
    for row in rows:
        ident = row["trajectory_id"]
        if not row["valid"]:
            results.append(dict(trajectory_id=ident, valid=False, reward=config.failure_reward,
                                failure_reason=row["failure_reason"], gate=gate))
            continue
        quality = scores[ident]["kfr"]["f1"]
        planning = scores[ident]["par"]["f1"]
        cost = row["extra_tool_calls"] / config.observation_budget
        penalty = float(gate) * (1 - quality) * cost
        results.append(dict(trajectory_id=ident, valid=True, **scores[ident], gate=gate,
            normalized_cost=cost, utcp=penalty,
            reward=config.alpha * quality + config.beta * planning - config.mu * penalty))
    return results
