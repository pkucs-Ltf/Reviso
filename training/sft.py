"""Minimal Qwen3.5 LoRA SFT over caller-supplied encoded inputs."""
import argparse
from pathlib import Path

from common.config import check_output_path, load_config, load_extension, required
from training.token_loss import make_labels, sparse_targets, supervised_token_loss


def collate_one(records):
    import torch

    if len(records) != 1:
        raise ValueError("Use one sequence per device; scale with gradient accumulation")
    batch = dict(records[0])
    if "labels" in batch:
        raise ValueError("Supply assistant_mask; labels are built by the public loss code")
    for key in ("input_ids", "assistant_mask", "attention_mask"):
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            raise ValueError(f"The input adapter must supply tensor {key}")
        if batch[key].ndim == 1:
            batch[key] = batch[key].unsqueeze(0)
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("Each dataset item must represent one sequence")
    batch["labels"] = make_labels(batch["input_ids"], batch.pop("assistant_mask"), batch["attention_mask"])
    return batch


def trainer_class():
    from transformers import Trainer

    class TokenTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The objective is a sequence mean, not a batch token-count ratio.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            payload = dict(inputs)
            labels = payload.pop("labels")
            positions, targets = sparse_targets(labels)
            if "logits_to_keep" in payload or "past_key_values" in payload:
                raise ValueError("Dataset inputs cannot override loss alignment or history")
            outputs = model(**payload, use_cache=False, logits_to_keep=positions)
            loss = supervised_token_loss(outputs.logits, targets)
            if not loss.isfinite():
                raise ValueError("Nonfinite SFT loss")
            return (loss, outputs) if return_outputs else loss

    return TokenTrainer


def run(config, *, resume=None):
    from common.model import load_qwen35
    from transformers import TrainingArguments, set_seed

    factory = load_extension(required(config, "data_factory"))
    model_path = required(config, "model_path")
    output = check_output_path(required(config, "output_dir"), model_path, config.get("adapter_path"))
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError("Output is nonempty; explicitly resume or choose a new directory")
    options = dict(config.get("trainer", {}))
    if options.get("per_device_train_batch_size", 1) != 1:
        raise ValueError("per_device_train_batch_size must be 1")
    if options.get("remove_unused_columns", False):
        raise ValueError("remove_unused_columns must be false for the token mask")
    if options.get("label_smoothing_factor", 0.0) != 0.0:
        raise ValueError("Label smoothing is not part of this objective")
    reserved = {"output_dir", "bf16", "fp16", "report_to"}
    if reserved & options.keys():
        raise ValueError(f"Set {sorted(reserved)} through the top-level configuration")
    set_seed(int(options.get("seed", 42)))
    model, processor = load_qwen35(model_path, adapter_path=config.get("adapter_path"),
        training=True, lora=config.get("lora"), dtype=config.get("dtype", "bfloat16"))
    dataset = factory(config=config, processor=processor)
    if not hasattr(dataset, "__len__") or len(dataset) == 0:
        raise ValueError("The input factory must return a nonempty map-style dataset")
    options.update(per_device_train_batch_size=1, remove_unused_columns=False)
    args = TrainingArguments(output_dir=str(output), report_to=[],
        bf16=config.get("dtype", "bfloat16") == "bfloat16", fp16=False, **options)
    trainer = trainer_class()(model=model, args=args, train_dataset=dataset,
                              data_collator=collate_one, processing_class=processor)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / "adapter"))
    trainer.save_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--data-factory")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Explicit framework checkpoint directory")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("model_path", "adapter_path", "data_factory", "output_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run(config, resume=args.resume)


if __name__ == "__main__":
    main()
