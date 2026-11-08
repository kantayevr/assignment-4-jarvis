"""Generate and log answers without reading the answer key."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from conditions import prepare_condition
from protocol import CONDITIONS, INFERENCE_FILES, ROOT, SETTINGS, data_directory, load_data, sha256, write_json


MODEL_FILE_SHA256 = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_model(model_dir, metadata, metadata_path):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_dir is None:
        model_dir = Path(snapshot_download(
            SETTINGS["model_id"], revision=SETTINGS["model_revision"], allow_patterns=list(MODEL_FILE_SHA256),
            cache_dir=ROOT / ".model-cache", token=False,
        ))
    model_hashes = {name: sha256(model_dir / name) for name in MODEL_FILE_SHA256}
    for name, expected in MODEL_FILE_SHA256.items():
        if model_hashes[name] != expected:
            raise ValueError(f"{name} does not match the pinned Qwen snapshot.")
    metadata["model_files_sha256"] = model_hashes
    metadata["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "safetensors", "jinja2")
    }
    write_json(metadata_path, metadata)
    torch.manual_seed(SETTINGS["seed"])
    torch.set_num_threads(SETTINGS["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation=SETTINGS["attention"],
    ).to(SETTINGS["device"]).eval()
    return torch, tokenizer, model


def predict(question, condition, snapshots, torch, tokenizer, model):
    start = time.perf_counter()
    retrieved, messages = prepare_condition(condition, question["question"], snapshots)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(SETTINGS["device"])
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens > SETTINGS["max_input_tokens"]:
        raise ValueError(f"Input exceeds the {SETTINGS['max_input_tokens']} token protocol limit")
    with torch.inference_mode():
        result = model.generate(
            **inputs, do_sample=SETTINGS["do_sample"], max_new_tokens=SETTINGS["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = result[0, input_tokens:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        **question, "condition": condition, "answer": answer,
        "retrieved": retrieved, "messages": messages, "input_tokens": input_tokens,
        "output_tokens": len(generated), "elapsed_seconds": time.perf_counter() - start,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def run(args, directory):
    data, hashes = load_data(directory, args.dataset, INFERENCE_FILES)
    questions = data["questions.json"]
    snapshots = {"frozen": data["kb_frozen.json"], "updated": data["kb_updated.json"]}
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Expected a nonempty set of distinct question IDs.")
    if args.dataset == "holdout" and len(questions) != 34:
        raise ValueError("The holdout must contain the 34 questions recorded in Appendix A.")
    for snapshot in snapshots.values():
        if len(snapshot) < SETTINGS["top_k"] or len({doc["document_id"] for doc in snapshot}) != len(snapshot):
            raise ValueError("Each snapshot must have at least three distinct documents.")
    random.Random(SETTINGS["seed"]).shuffle(questions)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata_path = args.output / "run_metadata.json"
    prediction_path = args.output / "predictions.jsonl"
    metadata = {
        "dataset": args.dataset, "status": "running", "started_at_utc": utc_now(),
        "settings": SETTINGS, "platform": platform.platform(), "python": platform.python_version(),
        "conditions": CONDITIONS, "question_order": [question["id"] for question in questions],
        "inputs_sha256": hashes,
        "source_sha256": {
            name: sha256(ROOT / "src" / name)
            for name in ("protocol.py", "rag.py", "conditions.py", "run_experiment.py")
        },
        "n_questions": len(questions), "n_predictions": 0,
    }
    write_json(metadata_path, metadata)
    stage = {"stage": "model_load"}
    try:
        print(f"Loading {SETTINGS['model_id']} at {SETTINGS['model_revision']}", flush=True)
        torch, tokenizer, model = load_model(args.model_dir, metadata, metadata_path)
        with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
            for index, question in enumerate(questions):
                offset = index % len(CONDITIONS)
                for condition in CONDITIONS[offset:] + CONDITIONS[:offset]:
                    stage = {"stage": "inference", "id": question["id"], "condition": condition}
                    row = predict(question, condition, snapshots, torch, tokenizer, model)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    metadata["n_predictions"] += 1
                    print(question["id"], condition, repr(row["answer"]), flush=True)
        stage = {"stage": "finalizing"}
        metadata["predictions_sha256"] = sha256(prediction_path)
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", ended_at_utc=utc_now(), failure={
            **stage, "type": type(error).__name__, "message": str(error),
        })
        write_json(metadata_path, metadata)
        raise
    metadata.update(status="completed", ended_at_utc=utc_now())
    write_json(metadata_path, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "holdout"), default="pilot")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, help="Local files from the pinned Qwen snapshot")
    parser.add_argument("--output", type=Path, required=True, help="A new directory for run evidence")
    args = parser.parse_args()
    run(args, data_directory(parser, args))


if __name__ == "__main__":
    main()
