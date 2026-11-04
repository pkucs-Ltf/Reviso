"""Qwen3.5-only loading. Model directories are read-only inputs."""
import os
from pathlib import Path


LORA_TARGETS = r".*language_model.*\.(q_proj|v_proj|in_proj_qkv)$"


def configure_cache():
    # Caches never default to the model or repository directory.
    root = Path(os.environ.get("REVISO_CACHE_DIR", "~/.cache/reviso")).expanduser()
    for name, suffix in (("HF_HOME", "hf"), ("TORCH_HOME", "torch"),
                         ("TRITON_CACHE_DIR", "triton")):
        os.environ.setdefault(name, str(root / suffix))


def load_qwen35(model_path, *, adapter_path=None, training=False, lora=None,
                dtype="bfloat16", attn_implementation="sdpa"):
    configure_cache()
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("Provide a local Qwen3.5 model directory")
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3_5":
        raise ValueError("This release only supports dense Qwen3.5")
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("dtype must be bfloat16 or float32")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, dtype), attn_implementation=attn_implementation,
    )
    model.requires_grad_(False)
    if adapter_path:
        from peft import PeftModel

        adapter = Path(adapter_path).expanduser().resolve()
        if not adapter.is_dir():
            raise ValueError("Provide a local LoRA directory")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training,
                                          local_files_only=True)
    elif training:
        from peft import LoraConfig, get_peft_model

        options = dict(lora or {})
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
            r=options.get("rank", 4), lora_alpha=options.get("alpha", 8),
            lora_dropout=options.get("dropout", 0.0), bias="none",
        ))
    if training:
        trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not trainable or any("lora_" not in name or "language_model" not in name
                                for name, _ in trainable):
            raise ValueError("Only language-side LoRA parameters may be trainable")
        for _, param in trainable:
            param.data = param.data.float()
        model.config.use_cache = False
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model, processor
