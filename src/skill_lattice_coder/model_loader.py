from pathlib import Path

from .config import base_config


def load_model(adapter: Path | None = None):
    from mlx_lm import load

    return load(
        base_config()["model"],
        adapter_path=str(adapter) if adapter else None,
    )


def generate_text(model, tokenizer, prompt: str) -> tuple[str, int, int]:
    from mlx_lm import generate

    config = base_config()
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=config["max_tokens"],
        temp=config["temperature"],
        verbose=False,
    )
    return output, len(tokenizer.encode(formatted)), len(tokenizer.encode(output))
