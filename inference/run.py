"""Qwen3.5 generation with an explicit, externally supplied observation backend."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from common.config import check_output_path, load_config, load_extension, required
from inference.prompts import SYSTEM_PROMPT
from inference.tool_interface import RecoverableToolError


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value

    def constant(value):
        raise ValueError("Nonfinite JSON number: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def unfence(text):
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        header, separator, rest = stripped.partition("\n")
        if separator and header in ("```", "```json"):
            return rest[:-3].strip()
    return stripped


def parse_final_sections(text):
    """Extract the caption from readable sections ending in a single STOP line."""
    captions = list(re.finditer(r"(?m)^[ \t]*final_caption[ \t]*:", text))
    actions = list(re.finditer(r"(?m)^[ \t]*action[ \t]*:[^\n]*", text))
    if len(captions) != 1 or len(actions) != 1:
        raise ValueError("Expected one final_caption section and one action: STOP line")
    caption_start, action = captions[0], actions[0]
    if (action.start() < caption_start.end()
            or not re.fullmatch(r"[ \t]*action[ \t]*:[ \t]*STOP[ \t\r]*", action.group())
            or text[action.end():].strip()):
        raise ValueError("The response must end with action: STOP after final_caption")
    caption = text[caption_start.end():action.start()].strip()
    if not caption:
        raise ValueError("Final caption must be nonempty")
    if re.search(r"(?m)^[ \t]*(?:local_caption|coarse_caption|corrected_coarse_caption|"
                 r"cfp_plan|viewing_plan(?:_update)?|execute_planned_action|remaining_actions)[ \t]*:",
                 caption):
        raise ValueError("Observation and plan sections must precede final_caption")
    return dict(state="stop", final_caption=caption)


def parse_reply(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty assistant response")
    if any(marker in text for marker in ("<|im_start|>", "<|im_end|>",
                                         "<tool_response>", "</tool_response>")):
        raise ValueError("Assistant authored a chat boundary or tool result")
    start, end = "<tool_call>", "</tool_call>"
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
            raise ValueError("Exactly one complete tool request is required")
        prefix, body = text.split(start)
        payload, suffix = body.split(end)
        if suffix.strip():
            raise ValueError("The assistant must yield immediately after its tool request")
        if ('"final_caption"' in prefix or '"STOP"' in prefix
                or re.search(r"(?m)^[ \t]*(?:final_caption|action)[ \t]*:", prefix)):
            raise ValueError("A final answer cannot be combined with a tool request")
        call = strict_json(unfence(payload))
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not call["name"].strip()
                or not isinstance(call["arguments"], dict)):
            raise ValueError("Malformed tool request")
        return dict(state="tool", call=call)
    final = unfence(text)
    if not final.startswith(("{", "[")):
        return parse_final_sections(final)
    # Retain compatibility with existing checkpoints that return JSON-only finals.
    result = strict_json(final)
    if (not isinstance(result, dict) or result.get("action") != "STOP"
            or not isinstance(result.get("final_caption"), str) or not result["final_caption"].strip()):
        raise ValueError("Expected final_caption and action=STOP")
    return dict(state="stop", final_caption=result["final_caption"])


def validate_backend(backend):
    for method in ("tool_specs", "initial_content", "encode", "execute"):
        if not callable(getattr(backend, method, None)):
            raise TypeError("Observation backend is missing " + method)


def tool_names(specs):
    if not isinstance(specs, list):
        raise ValueError("tool_specs must return a list")
    names = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each tool schema must be a mapping")
        function = spec.get("function", {})
        if not isinstance(function, dict):
            raise ValueError("function must be a mapping")
        name = function.get("name")
        if spec.get("type") != "function" or not isinstance(name, str) or not name.strip():
            raise ValueError("Tools require type=function and a nonempty function.name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tool names")
    return set(names)


def content_present(content):
    return (isinstance(content, str) and bool(content.strip())) or (isinstance(content, list) and bool(content))


def run_episode(*, generate, backend, initial_content, specs, max_turns):
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("max_turns must be a positive integer")
    if not content_present(initial_content):
        raise ValueError("The initial observation is empty")
    allowed = tool_names(specs)
    history = [dict(role="system", content=SYSTEM_PROMPT),
               dict(role="user", content=deepcopy(initial_content))]
    assistant_turns, requests, tool_errors = [], [], []
    for turn in range(max_turns):
        final_turn = turn == max_turns - 1
        if final_turn:
            history.append(dict(role="user", content=(
                "This is the final response turn. No further tool calls are available. "
                "Interpret the latest observation or error, clear remaining actions with "
                "viewing_plan_update and a reason, and provide final_caption followed by "
                "action: STOP. Use only collected evidence and omit or qualify unresolved "
                "details. Do not claim that a failed check succeeded."
            )))
        # Extension functions receive copies and cannot rewrite previous evidence.
        text = generate(deepcopy(history), [] if final_turn else deepcopy(specs))
        parsed = parse_reply(text)
        history.append(dict(role="assistant", content=text))
        assistant_turns.append(text)
        if parsed["state"] == "stop":
            return dict(caption=parsed["final_caption"], assistant_turns=assistant_turns,
                        tool_calls=requests, extra_tool_calls=len(requests), tool_errors=tool_errors)
        if final_turn:
            raise RuntimeError("The final response turn cannot execute another tool call")
        call = parsed["call"]
        if call["name"] not in allowed:
            raise ValueError("The assistant requested an unavailable tool")
        requests.append(deepcopy(call))
        try:
            observation = backend.execute(name=call["name"], arguments=deepcopy(call["arguments"]))
            if observation is None or (isinstance(observation, (str, list))
                                       and not content_present(observation)):
                raise RecoverableToolError("The tool returned no usable observation.")
            if not content_present(observation):
                raise TypeError("Tool observations must be a string or a list of content blocks")
        except RecoverableToolError as exc:
            error = dict(tool=call["name"], message=str(exc).strip() or "The observation request failed.",
                         recoverable=True)
            tool_errors.append(dict(call_index=len(requests) - 1, **error))
            observation = json.dumps(dict(status="error", error=error), ensure_ascii=False)
        history.append(dict(role="tool", name=call["name"], content=deepcopy(observation)))
    raise RuntimeError("Generation stopped at the turn limit without a final caption")


def make_generator(model, processor, backend, config):
    import torch

    device = torch.device(config.get("device", "cuda:0"))
    model.to(device).eval()
    tokenizer = processor.tokenizer
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError("Missing Qwen3.5 end-of-turn token")
    options = dict(config.get("generation", {}))
    if set(options) - {"max_new_tokens", "do_sample", "temperature", "top_p"}:
        raise ValueError("Unsupported generation option")
    limit = options.get("max_new_tokens")
    if type(limit) is not int or limit <= 0:
        raise ValueError("Set a positive max_new_tokens")
    if not options.get("do_sample", False):
        options.pop("temperature", None)
        options.pop("top_p", None)

    def generate(history, specs):
        batch = backend.encode(messages=history, tools=specs, processor=processor)
        if not isinstance(batch, dict) or any(k in batch for k in
                ("labels", "assistant_mask", "past_key_values", "logits_to_keep")):
            raise ValueError("Encoding must contain only current-history model inputs")
        ids = batch.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Inference expects one encoded conversation")
        length = ids.shape[1]
        context_limit = model.config.text_config.max_position_embeddings
        if length + limit > context_limit:
            raise ValueError("Context budget exceeded; history is never silently truncated")
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in batch.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, **options, eos_token_id=eos,
                                    pad_token_id=tokenizer.pad_token_id)
        tokens = output[0, length:].tolist()
        if not tokens or tokens[-1] != eos:
            raise RuntimeError("Incomplete assistant turn; generation ended without its boundary")
        return tokenizer.decode(tokens[:-1], skip_special_tokens=False).strip()

    return generate


def run(config, source):
    from common.model import load_qwen35

    factory = load_extension(required(config, "backend"))
    model_path = required(config, "model_path")
    output = check_output_path(required(config, "output_file"), model_path,
                              config.get("adapter_path"), source)
    if output.exists():
        raise FileExistsError("Choose a new output file; existing results are not overwritten")
    backend = factory(config=config)
    validate_backend(backend)
    specs = backend.tool_specs()
    tool_names(specs)
    initial = backend.initial_content(source)
    if not content_present(initial):
        raise ValueError("The initial observation is empty")
    model, processor = load_qwen35(model_path, adapter_path=config.get("adapter_path"),
                                   dtype=config.get("dtype", "bfloat16"))
    result = run_episode(generate=make_generator(model, processor, backend, config), backend=backend,
        initial_content=initial, specs=specs, max_turns=required(config, "max_turns"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["caption"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--input", required=True, help="Caller-owned video/input path")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--backend", help="Observation backend factory as module:callable")
    parser.add_argument("--output-file")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("model_path", "adapter_path", "backend", "output_file"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run(config, args.input)


if __name__ == "__main__":
    main()
