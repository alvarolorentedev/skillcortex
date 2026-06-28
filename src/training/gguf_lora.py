from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    train_from_config(Path(args.config))
    return 0


def train_from_config(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["adapter_path"])
    peft_output = output / "peft"
    output.mkdir(parents=True, exist_ok=True)
    _train_peft_lora(config, peft_output)
    converter = config.get("gguf_converter")
    if not converter:
        raise ValueError("GGUF training requires gguf_converter in base config")
    subprocess.run(
        [
            "python",
            str(converter),
            str(peft_output),
            "--outfile",
            str(output / "adapter.gguf"),
        ],
        check=True,
    )


def _train_peft_lora(config: dict, output: Path) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        raise ImportError("GGUF training requires the slmcortex[gguf] extra") from error

    tokenizer = AutoTokenizer.from_pretrained(config["source_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["source_model"])
    lora = config["lora_parameters"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=float(lora["scale"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["keys"]),
            task_type="CAUSAL_LM",
        ),
    )

    class JsonlDataset(Dataset):
        def __init__(self, path: Path):
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            messages = self.rows[index]["messages"]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                text = "\n".join(message["content"] for message in messages)
            encoded = tokenizer(text, truncation=True, max_length=2048)
            encoded["labels"] = list(encoded["input_ids"])
            return {key: torch.tensor(value) for key, value in encoded.items()}

    train = JsonlDataset(Path(config["data"]) / "train.jsonl")
    args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=int(config["batch_size"]),
        max_steps=int(config["iters"]),
        learning_rate=float(config["learning_rate"]),
        report_to=[],
        save_strategy="no",
        seed=int(config["seed"]),
    )
    Trainer(model=model, args=args, train_dataset=train).train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    raise SystemExit(main())
