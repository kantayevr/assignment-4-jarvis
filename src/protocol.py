"""Frozen experiment settings and dataset identity checks."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "pilot"
HOLDOUT_MANIFEST_SHA256 = "efef89519962197ed1d748ed99c43ab90fcf23db6ef4be7e9f918f710b5ba788"
CONDITIONS = ("tfidf_frozen", "tfidf_updated", "closed_book", "bm25_updated")
RETRIEVAL = {"tfidf_frozen": "frozen", "tfidf_updated": "updated", "bm25_updated": "updated"}
INFERENCE_FILES = ("questions.json", "kb_frozen.json", "kb_updated.json")
SETTINGS = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "device": "cpu",
    "dtype": "float32",
    "attention": "eager",
    "threads": 4,
    "seed": 42,
    "top_k": 3,
    "max_input_tokens": 2048,
    "max_new_tokens": 32,
    "do_sample": False,
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_data(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def data_directory(parser, args):
    if args.dataset == "pilot":
        if args.data_dir is not None:
            parser.error("The pilot uses the bundled Assignment 3 files. Omit --data-dir.")
        return PILOT
    if args.data_dir is None:
        parser.error("Provide the extracted dataset with --data-dir for the holdout.")
    return args.data_dir


def load_data(directory, dataset, names):
    payloads = {name: (directory / name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    if dataset == "holdout":
        manifest_bytes = (directory / "dataset_manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != HOLDOUT_MANIFEST_SHA256:
            raise ValueError("The dataset manifest does not match Appendix A.")
        expected = json.loads(manifest_bytes)["sha256"]
        for name, checksum in hashes.items():
            if expected[name] != checksum:
                raise ValueError(f"{name} does not match the frozen dataset manifest.")
        hashes["dataset_manifest.json"] = HOLDOUT_MANIFEST_SHA256
    return {name: json.loads(payload) for name, payload in payloads.items()}, hashes
