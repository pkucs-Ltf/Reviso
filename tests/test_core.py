"""CPU-only tests with synthetic in-memory values; no dataset fixtures."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from common.config import check_output_path
from inference.run import parse_reply, run_episode, tool_names, validate_backend
from inference.tool_interface import RecoverableToolError
from training.grpo import GroupRewardHook, terminal_rewards
from training.reward import JudgeError, RewardConfig, matching_score, reference_digest, score_group
from training.sft import collate_one
from training.token_loss import causal_loss, make_labels, sparse_targets, supervised_token_loss


class LossTests(unittest.TestCase):
    def test_labels_and_padding(self):
        ids = torch.arange(8).reshape(1, 8)
        mask = torch.tensor([[False, False, True, True, False, True, True, True]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        result = make_labels(ids, mask, attention)
        self.assertEqual(result.tolist(), [[-100, -100, 2, 3, -100, 5, -100, -100]])

    def test_sparse_dense_loss_and_gradient(self):
        torch.manual_seed(7)
        ids = torch.tensor([[1, 2, 3, 4, 2, 1]])
        mask = torch.tensor([[False, False, True, True, False, True]])
        labels = make_labels(ids, mask)
        dense = torch.randn(1, 6, 7, requires_grad=True)
        sparse = dense.detach().clone().requires_grad_()
        positions, targets = sparse_targets(labels)
        a = causal_loss(dense, labels, chunk_size=2)
        b = supervised_token_loss(sparse[:, positions], targets, chunk_size=2)
        self.assertTrue(torch.allclose(a, b))
        a.backward()
        b.backward()
        self.assertTrue(torch.allclose(dense.grad, sparse.grad))
        self.assertEqual(float(dense.grad[0, 0].abs().sum()), 0.0)

    def test_equal_sequence_weight_not_pooled_tokens(self):
        logits = torch.tensor([[[4., 0.], [4., 0.], [4., 0.]],
                               [[0., 0.], [0., 0.], [0., 0.]]])
        targets = torch.tensor([[0, -100, -100], [0, 0, 0]])
        result = supervised_token_loss(logits, targets)
        expected = (F.cross_entropy(logits[0, :1], targets[0, :1])
                    + F.cross_entropy(logits[1], targets[1])) / 2
        pooled = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(result, expected))
        self.assertFalse(torch.allclose(result, pooled))

    def test_invalid_masks(self):
        ids = torch.arange(3).reshape(1, 3)
        for mask in (torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids, dtype=torch.bool), ids):
            with self.assertRaises(ValueError):
                make_labels(ids, mask)

    def test_collator_keeps_model_tensors(self):
        visual = torch.randn(3, 4)
        item = dict(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.ones(3, dtype=torch.long),
                    assistant_mask=torch.tensor([False, False, True]), pixel_values=visual)
        batch = collate_one([item])
        self.assertIs(batch["pixel_values"], visual)
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 3]])
        self.assertIn("assistant_mask", item)
        with self.assertRaises(ValueError):
            collate_one([item, item])


def abstract_group():
    # Symbols exercise graph/counting logic; they are not caption examples.
    reference = dict(id="reference", facts=[dict(id="f0", text="A"), dict(id="f1", text="B")],
                     plan_points=[dict(id="p0", text="C")])
    rows = [dict(trajectory_id=str(i), group_id="group", policy_version="policy",
                 reference_id=reference["id"], reference_digest=reference_digest(reference),
                 observation_digest="observation", valid=True, extra_tool_calls=1,
                 final_caption="A B", initial_plan="C") for i in range(2)]
    verdict = dict(status="ok", caption=dict(
        generated=[dict(id="a", text="A", source_quote="A"), dict(id="b", text="B", source_quote="B")],
        matches=[dict(generated_id="a", reference_id="f0", reason="equal"),
                 dict(generated_id="b", reference_id="f1", reason="equal")]),
        plan=dict(generated=[dict(id="c", text="C", source_quote="C")],
                  matches=[dict(generated_id="c", reference_id="p0", reason="equal")]))
    return rows, reference, {row["trajectory_id"]: copy.deepcopy(verdict) for row in rows}


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.reference, self.verdicts = abstract_group()
        self.config = RewardConfig(0.7, 10, group_size=2)

    def score(self):
        return score_group(self.rows, self.reference, self.verdicts, self.config)

    def test_perfect(self):
        result = self.score()[0]
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["par"]["f1"], 1.0)
        self.assertEqual(result["utcp"], 0.0)
        self.assertNotIn("mfar", result)
        self.assertNotIn("qgop", result)

    def test_group_gate(self):
        self.verdicts["1"]["caption"]["matches"] = []
        self.rows[1]["extra_tool_calls"] = 10
        self.assertAlmostEqual(self.score()[1]["reward"], 0.15)
        self.verdicts["0"]["caption"]["matches"] = []
        self.assertAlmostEqual(self.score()[1]["reward"], 0.2)

    def test_zero_calls_have_zero_cost_with_active_gate(self):
        self.verdicts["1"]["caption"]["matches"] = self.verdicts["1"]["caption"]["matches"][:1]
        self.rows[1]["extra_tool_calls"] = 0
        result = self.score()[1]
        self.assertTrue(result["gate"])
        self.assertEqual(result["kfr"]["f1"], 0.5)
        self.assertEqual(result["normalized_cost"], 0.0)
        self.assertEqual(result["utcp"], 0.0)
        self.assertAlmostEqual(result["reward"], 0.6)

    def test_more_calls_increase_penalty_at_same_quality(self):
        self.verdicts["1"]["caption"]["matches"] = self.verdicts["1"]["caption"]["matches"][:1]
        fewer_calls = self.score()[1]
        self.rows[1]["extra_tool_calls"] = 4
        more_calls = self.score()[1]
        self.assertTrue(fewer_calls["gate"])
        self.assertTrue(more_calls["gate"])
        self.assertEqual(fewer_calls["kfr"]["f1"], more_calls["kfr"]["f1"])
        self.assertAlmostEqual(fewer_calls["normalized_cost"], 0.1)
        self.assertAlmostEqual(more_calls["normalized_cost"], 0.4)
        self.assertAlmostEqual(fewer_calls["utcp"], 0.05)
        self.assertAlmostEqual(more_calls["utcp"], 0.2)
        self.assertLess(more_calls["reward"], fewer_calls["reward"])

    def test_missing_call_count_and_legacy_token_cost_are_rejected(self):
        self.rows[1].pop("extra_tool_calls")
        with self.assertRaises(ValueError):
            self.score()
        self.rows[1]["extra_visual_tokens"] = 1
        with self.assertRaises(ValueError):
            self.score()

    def test_duplicate_edges_and_one_to_one(self):
        section = self.verdicts["0"]["caption"]
        section["matches"] *= 3
        self.assertEqual(self.score()[0]["kfr"]["f1"], 1.0)
        for edge in section["matches"]:
            edge["reference_id"] = "f0"
        self.assertEqual(self.score()[0]["kfr"]["f1"], 0.5)

    def test_incomplete_and_mixed_groups(self):
        with self.assertRaises(ValueError):
            score_group(self.rows[:1], self.reference, self.verdicts, self.config)
        self.rows[1]["policy_version"] = "different"
        with self.assertRaises(ValueError):
            self.score()

    def test_reference_fingerprint(self):
        self.reference["facts"][0]["text"] = "different"
        with self.assertRaises(ValueError):
            self.score()

    def test_missing_and_uncertain_judgment(self):
        self.verdicts.pop("1")
        with self.assertRaises(JudgeError):
            self.score()
        self.verdicts["1"] = dict(status="uncertain")
        with self.assertRaises(JudgeError):
            self.score()

    def test_wrong_source_quote(self):
        self.verdicts["0"]["caption"]["generated"][0]["source_quote"] = "missing"
        with self.assertRaises(JudgeError):
            self.score()

    def test_policy_failure_versus_technical_failure(self):
        self.rows[0].update(valid=False, failure_reason="invalid_output")
        del self.verdicts["0"]
        self.assertEqual(self.score()[0]["reward"], -1.0)
        self.rows[0]["technical_failure"] = True
        with self.assertRaises(JudgeError):
            self.score()

    def test_empty_claims_have_zero_f1(self):
        value = matching_score(dict(generated=[], matches=[]), self.reference["facts"], "")
        self.assertEqual(value["f1"], 0.0)

    def test_judge_hook_and_terminal_token_alignment(self):
        hook = GroupRewardHook(vars(self.config))
        judge = Mock(side_effect=lambda **kw: self.verdicts[kw["rollout"]["trajectory_id"]])
        scores = hook(self.rows, self.reference, judge=judge)
        self.assertEqual(judge.call_count, 2)
        mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
        result = terminal_rewards(mask, scores, ["1", "0"])
        self.assertEqual(result.tolist(), [[0., 0., 1., 0.], [0., 1., 0., 0.]])
        with self.assertRaises(ValueError):
            terminal_rewards(mask, scores, ["0", "0"])


class InferenceTests(unittest.TestCase):
    def test_readable_multiline_final_and_json_compatibility(self):
        caption = "The person lifts the lid.\nThe contents remain obscured."
        readable = "local_caption: The lid moves upward.\nfinal_caption: " + caption + "\naction: STOP"
        expected = dict(state="stop", final_caption=caption)
        self.assertEqual(parse_reply(readable), expected)
        self.assertEqual(parse_reply(json.dumps(dict(final_caption=caption, action="STOP"))), expected)

    def test_readable_final_requires_unambiguous_terminal_stop(self):
        invalid = [
            "local_caption: visible detail\naction: STOP",
            "final_caption: caption",
            "final_caption: first\nfinal_caption: second\naction: STOP",
            "final_caption: caption\naction: STOP\naction: STOP",
            "final_caption: caption\naction: STOP\ntrailing content",
            "final_caption: caption\naction: CONTINUE",
            "final_caption: \naction: STOP",
            'final_caption: caption\n<tool_call>{"name":"inspect","arguments":{}}</tool_call>',
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_reply(value)

    def test_full_history_and_real_tool_result(self):
        backend = Mock()
        backend.execute.return_value = "observation"
        specs = [dict(type="function", function=dict(name="inspect", parameters={}))]
        seen = []

        def generate(history, tools):
            seen.append(copy.deepcopy(history))
            if len(seen) == 1:
                return 'Plan. <tool_call>{"name":"inspect","arguments":{}}</tool_call>'
            return '{"final_caption":"result","action":"STOP"}'

        result = run_episode(generate=generate, backend=backend, initial_content="initial",
                             specs=specs, max_turns=3)
        self.assertEqual(result["caption"], "result")
        self.assertEqual([m["role"] for m in seen[1]], ["system", "user", "assistant", "tool"])
        self.assertEqual(seen[1][-1]["content"], "observation")
        backend.execute.assert_called_once_with(name="inspect", arguments={})

    def test_recoverable_tool_failure_allows_revised_request(self):
        backend = Mock()
        backend.execute.side_effect = [RecoverableToolError("start must be before end"), "observation"]
        specs = [dict(type="function", function=dict(name="inspect", parameters={}))]
        replies = [
            '<tool_call>{"name":"inspect","arguments":{"start":9,"end":3}}</tool_call>',
            '<tool_call>{"name":"inspect","arguments":{"start":3,"end":9}}</tool_call>',
            "final_caption: supported caption\naction: STOP",
        ]
        seen = []

        def generate(history, tools):
            seen.append(copy.deepcopy(history))
            return replies[len(seen) - 1]

        result = run_episode(generate=generate, backend=backend, initial_content="initial",
                             specs=specs, max_turns=4)
        failed_observation = seen[1][-1]
        self.assertEqual(failed_observation["role"], "tool")
        error = json.loads(failed_observation["content"])
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["error"]["message"], "start must be before end")
        self.assertIn(failed_observation, seen[2])
        self.assertEqual(seen[2][-1]["content"], "observation")
        self.assertEqual([call.kwargs["arguments"] for call in backend.execute.call_args_list],
                         [dict(start=9, end=3), dict(start=3, end=9)])
        self.assertEqual(result["caption"], "supported caption")
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertEqual(result["extra_tool_calls"], 2)
        self.assertEqual(len(result["tool_errors"]), 1)
        self.assertEqual(len(result["assistant_turns"]), 3)

    def test_empty_observation_allows_conservative_final(self):
        backend = Mock()
        backend.execute.return_value = ""
        specs = [dict(type="function", function=dict(name="inspect"))]
        seen = []

        def generate(history, tools):
            seen.append(copy.deepcopy(history))
            if len(seen) == 1:
                return '<tool_call>{"name":"inspect","arguments":{}}</tool_call>'
            return "final_caption: A person stands beside a closed container.\naction: STOP"

        result = run_episode(generate=generate, backend=backend, initial_content="initial",
                             specs=specs, max_turns=3)
        self.assertEqual(seen[1][-1]["role"], "tool")
        error = json.loads(seen[1][-1]["content"])
        self.assertEqual(error["status"], "error")
        self.assertIn("observation", error["error"]["message"].lower())
        self.assertEqual(result["caption"], "A person stands beside a closed container.")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["extra_tool_calls"], 1)
        self.assertEqual(len(result["tool_errors"]), 1)

    def test_last_turn_preserves_evidence_and_requests_final_without_tools(self):
        backend = Mock()
        backend.execute.return_value = "observation"
        specs = [dict(type="function", function=dict(name="inspect"))]
        seen = []

        def generate(history, tools):
            seen.append((copy.deepcopy(history), copy.deepcopy(tools)))
            if len(seen) == 1:
                return '<tool_call>{"name":"inspect","arguments":{}}</tool_call>'
            return "final_caption: supported caption\naction: STOP"

        result = run_episode(generate=generate, backend=backend, initial_content="initial",
                             specs=specs, max_turns=2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][1], specs)
        self.assertEqual(seen[1][1], [])
        final_history = seen[1][0]
        self.assertEqual(final_history[-2]["role"], "tool")
        self.assertEqual(final_history[-2]["content"], "observation")
        self.assertEqual(final_history[-1]["role"], "user")
        self.assertIn("final", final_history[-1]["content"].lower())
        self.assertEqual(result["caption"], "supported caption")
        self.assertEqual(result["extra_tool_calls"], 1)
        backend.execute.assert_called_once()

    def test_last_turn_tool_request_is_rejected_without_execution(self):
        backend = Mock()
        backend.execute.return_value = "observation"
        specs = [dict(type="function", function=dict(name="inspect"))]
        generate = Mock(return_value='<tool_call>{"name":"inspect","arguments":{}}</tool_call>')
        with self.assertRaises((ValueError, RuntimeError)):
            run_episode(generate=generate, backend=backend, initial_content="initial",
                        specs=specs, max_turns=2)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args.args[1], [])
        backend.execute.assert_called_once()

    def test_single_turn_budget_finalizes_without_observation_call(self):
        backend = Mock()
        specs = [dict(type="function", function=dict(name="inspect"))]
        generate = Mock(return_value="final_caption: preview-supported caption\naction: STOP")
        result = run_episode(generate=generate, backend=backend, initial_content="initial",
                             specs=specs, max_turns=1)
        generate.assert_called_once()
        history, supplied_tools = generate.call_args.args
        self.assertEqual(supplied_tools, [])
        self.assertEqual(history[-1]["role"], "user")
        self.assertIn("final", history[-1]["content"].lower())
        self.assertEqual(result["extra_tool_calls"], 0)
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["tool_errors"], [])
        backend.execute.assert_not_called()

    def test_bad_protocol_rejected(self):
        invalid = ['{}', '{"final_caption":"x","action":"STOP","action":"GO"}',
                   '<tool_response>invented</tool_response>',
                   '<tool_call>{"name":"x","arguments":{}}</tool_call>after',
                   '<tool_call>{"name":"x","arguments":{"x":NaN}}</tool_call>']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_reply(value)

    def test_unknown_tool_and_no_fallback(self):
        backend = Mock()
        with self.assertRaises(ValueError):
            run_episode(generate=lambda *_: '<tool_call>{"name":"x","arguments":{}}</tool_call>',
                        backend=backend, initial_content="initial", specs=[], max_turns=2)
        backend.execute.assert_not_called()

    def test_tool_failure_propagates(self):
        backend = Mock()
        backend.execute.side_effect = RuntimeError("unavailable")
        specs = [dict(type="function", function=dict(name="x"))]
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            run_episode(generate=lambda *_: '<tool_call>{"name":"x","arguments":{}}</tool_call>',
                        backend=backend, initial_content="initial", specs=specs, max_turns=2)

    def test_missing_backend_and_invalid_schema(self):
        with self.assertRaises(TypeError):
            validate_backend(object())
        with self.assertRaises(ValueError):
            tool_names(["not a mapping"])

    def test_read_only_paths(self):
        model = Path("model").resolve()
        with self.assertRaises(ValueError):
            check_output_path(model / "output", model)
        self.assertEqual(check_output_path("outputs", model), Path("outputs").resolve())


if __name__ == "__main__":
    unittest.main()
