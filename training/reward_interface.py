"""Judge protocol only: no credentials, providers, requests or retry services."""
from typing import Protocol


JUDGE_INSTRUCTIONS = """Compare the supplied generated caption and initial plan with their
respective reference atoms. References are for evaluation only, not policy input.
Extract distinct generated atoms, including unsupported claims; do not omit a
claim merely because it has no reference match. Deduplicate semantic repetitions.
For each generated atom return id, text, and a verbatim source_quote from the
corresponding generated output. Match only semantically supported generated and
reference atoms. Return status='ok' with caption and plan sections, each containing
generated and matches lists. Each match must have generated_id, reference_id and
reason. Use an empty matches list for genuine nonmatches. If a reliable judgment
cannot be made, return status='uncertain', not invented evidence or zero scores.
Do not infer whether an unseen video supports a statement; this is text matching.
"""


class SemanticJudge(Protocol):
    def __call__(self, *, rollout: dict, reference: dict, instructions: str) -> dict:
        """Return evidence for this rollout only; use an independent request context."""
        ...


def judge_group(rows, reference, judge: SemanticJudge):
    """Propagate technical failures; never fabricate missing judgments."""
    verdicts = {}
    for row in rows:
        if row["valid"]:
            verdicts[row["trajectory_id"]] = judge(
                rollout=row, reference=reference, instructions=JUDGE_INSTRUCTIONS)
    return verdicts
