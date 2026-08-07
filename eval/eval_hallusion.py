"""Aligned Native and VisualMargin evaluation on HallusionBench."""

import argparse
import copy
import gc
import json
import os
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.visual_margin import generate_visual_margin

MASK_ID = 126336
BAD_IDS = [126081, 126080, 126346, 126347]
HALLUSION_IMAGE_DATA_FILE = (
    "https://huggingface.co/datasets/lmms-lab/HallusionBench/resolve/main/"
    "data/image-00000-of-00001.parquet"
)


def resize_image(image, max_side=384):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)


def parse_yes_no(text):
    matches = re.findall(
        r"<answer>\s*(yes|no)\s*</answer>", text, re.IGNORECASE | re.DOTALL
    )
    if matches:
        return matches[-1].lower()
    matches = re.findall(r"<answer>\s*(yes|no)", text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    normalized = text.strip().rstrip(".")
    if normalized.lower() in {"yes", "no"}:
        return normalized.lower()
    if normalized:
        first = normalized.split()[0].lower().rstrip(",.")
        if first in {"yes", "no"}:
            return first
    for word in reversed(normalized.split()):
        cleaned = word.strip().rstrip(".,;:!").lower()
        if cleaned in {"yes", "no"}:
            return cleaned
    return "unknown"


@torch.no_grad()
def generate_native(
    model,
    inputs_embeds,
    tokenizer,
    gen_length=256,
    block_length=256,
    steps=32,
    temperature=0.0,
    stopping_criteria=None,
):
    del temperature
    device = inputs_embeds.device
    backbone = model.model
    embed_tokens = backbone.embed_tokens
    prompt_length = inputs_embeds.shape[1]
    total_length = prompt_length + gen_length
    mask_embed = embed_tokens(torch.tensor([MASK_ID], device=device))
    x_embeds = mask_embed.repeat(1, total_length, 1)
    x_embeds[:, :prompt_length] = inputs_embeds
    x = torch.full((1, total_length), MASK_ID, dtype=torch.long, device=device)

    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by the number of blocks")
    steps_per_block = steps // num_blocks
    stop_position = total_length
    found_stop = False
    stop_tokens = [
        tokenizer.encode(stop, add_special_tokens=False)
        for stop in (stopping_criteria or [])
    ]

    for block_index in range(num_blocks):
        block_start = prompt_length + block_index * block_length
        block_end = block_start + block_length
        block_mask = torch.all(
            torch.abs(x_embeds[:, block_start:block_end] - mask_embed) < 1e-5,
            dim=2,
        )
        transfer_quota = model.get_num_transfer_tokens(
            block_mask, steps_per_block
        )
        for step_index in range(steps_per_block):
            gc.collect()
            torch.cuda.empty_cache()
            mask_index = torch.all(
                torch.abs(x_embeds - mask_embed) < 1e-5, dim=2
            )
            if found_stop and not mask_index[0, prompt_length:stop_position].any():
                break
            if not mask_index[0, block_start:block_end].any():
                break

            masked = mask_index[0]
            with torch.cuda.amp.autocast(enabled=True):
                outputs = backbone(inputs_embeds=x_embeds)
            logits = model.lm_head(outputs.last_hidden_state[0, masked]).float()
            for token_id in BAD_IDS:
                logits[:, token_id] = -float("inf")
            candidate_ids = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)
            confidence = torch.gather(
                probabilities, 1, candidate_ids.unsqueeze(1)
            ).squeeze(1)

            candidates = x.clone()
            candidates[0, masked] = candidate_ids
            scores = torch.full(
                (1, total_length), -float("inf"), device=device
            )
            scores[0, masked] = confidence
            scores[:, block_end:] = -float("inf")
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            quota = int(transfer_quota[0, step_index].item())
            if quota:
                selected = torch.topk(scores[0], k=quota).indices
                transfer_index[0, selected] = True

            candidate_embeds = embed_tokens(candidates)
            x_embeds[transfer_index] = candidate_embeds[transfer_index]
            x[transfer_index] = candidates[transfer_index]

            if stop_tokens and not found_stop:
                generated = x[0, prompt_length : prompt_length + gen_length]
                for stop in stop_tokens:
                    stop_tensor = torch.tensor(stop, device=device)
                    for start in range(generated.size(0) - len(stop) + 1):
                        if torch.all(
                            generated[start : start + len(stop)] == stop_tensor
                        ):
                            stop_position = prompt_length + start
                            found_stop = True
                            break
                    if found_stop:
                        break

            del outputs, logits, probabilities, confidence, candidates
            del scores, transfer_index, candidate_embeds

    return tokenizer.decode(
        x[0, prompt_length:stop_position], skip_special_tokens=True
    )


def load_items(dataset_arrow=None):
    if dataset_arrow:
        dataset = Dataset.from_file(dataset_arrow)
    else:
        dataset = load_dataset(
            "parquet",
            data_files={"image": HALLUSION_IMAGE_DATA_FILE},
            split="image",
        )
    items = []
    for index, row in enumerate(dataset):
        ground_truth = str(row["gt_answer"]).strip()
        answer = "yes" if ground_truth in {"1", "yes", "Yes", "true", "True"} else "no"
        items.append(
            {
                "idx": index,
                "image": row["image"],
                "question": row["question"].strip(),
                "answer": answer,
                "category": row.get("category", "unknown"),
                "subcategory": row.get("subcategory", "unknown"),
                "set_id": str(row.get("set_id", "0")),
                "question_id": str(row.get("question_id", index)),
                "figure_id": str(row.get("figure_id", "0")),
                "gt_answer": ground_truth,
            }
        )
    return items


def score_hallusion(rows):
    for row in rows:
        row["correct"] = (
            "1" if row["pred"] == "yes" else "0"
        ) == row["gt_answer"]

    def grouped_accuracy(fields, skip_vs_nofig=False):
        groups = defaultdict(list)
        for row in rows:
            if skip_vs_nofig and row["category"] == "VS" and row["figure_id"] == "0":
                continue
            key = tuple(row[field] for field in fields)
            groups[key].append(row["correct"])
        values = [all(group) for group in groups.values()]
        return {
            "accuracy": sum(values) / len(values) if values else 0.0,
            "correct": int(sum(values)),
            "total": len(values),
        }

    correct = sum(row["correct"] for row in rows)
    return {
        "aAcc": {
            "accuracy": correct / len(rows) if rows else 0.0,
            "correct": int(correct),
            "total": len(rows),
        },
        "qAcc": grouped_accuracy(
            ["category", "subcategory", "set_id", "question_id"]
        ),
        "fAcc": grouped_accuracy(
            ["category", "subcategory", "set_id", "figure_id"],
            skip_vs_nofig=True,
        ),
    }


def make_prompt(question):
    return (
        "This is a yes-or-no visual question. Use the image and the question to answer carefully.\n"
        f"Question: {question}\n"
        "Rules: answer yes only if the statement is supported by the image; "
        "answer no if it is false, absent, uncertain, or not visually supported. "
        "The final answer must be exactly yes or no, not a sentence. "
        "End the response with exactly one final tag in this format: "
        "<answer>yes</answer> or <answer>no</answer>. /think"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["native", "visual_margin"], required=True)
    parser.add_argument("--pretrained", default="GSAI-ML/LLaDA-V")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_arrow")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        "llava_llada",
        attn_implementation="sdpa",
        device_map=args.device,
    )
    model.eval()
    items = load_items(args.dataset_arrow)
    items = items[: min(args.max_samples or len(items), len(items))]
    conversation = conv_templates["llava_llada"]
    stop_string = conversation.sep

    prediction_path = os.path.join(
        args.output_dir, f"hallusion_{args.method}_predictions.jsonl"
    )
    rows = []
    if args.resume and os.path.exists(prediction_path):
        with open(prediction_path) as existing_file:
            rows = [json.loads(line) for line in existing_file if line.strip()]
        completed = {row["idx"] for row in rows}
        items = [item for item in items if item["idx"] not in completed]

    oom = 0
    errors = 0
    started = time.time()
    mode = "a" if args.resume else "w"
    with open(prediction_path, mode) as prediction_file:
        for item in tqdm(items, desc=f"hallusion-{args.method}"):
            try:
                image = resize_image(item["image"].convert("RGB"))
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = [
                    tensor.to(dtype=torch.float16, device=args.device)
                    for tensor in image_tensor
                ]
                conv = copy.deepcopy(conversation)
                conv.append_message(
                    conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + make_prompt(item["question"])
                )
                conv.append_message(conv.roles[1], None)
                input_ids = tokenizer_image_token(
                    conv.get_prompt(),
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(args.device)
                _, _, _, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    None,
                    None,
                    None,
                    image_tensor,
                    ["image"],
                    image_sizes=[image.size],
                )

                if args.method == "visual_margin":
                    gray_image = Image.new("RGB", image.size, color=(128, 128, 128))
                    gray_tensor = process_images(
                        [gray_image], image_processor, model.config
                    )
                    gray_tensor = [
                        tensor.to(dtype=torch.float16, device=args.device)
                        for tensor in gray_tensor
                    ]
                    _, _, _, _, gray_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        None,
                        None,
                        None,
                        None,
                        gray_tensor,
                        ["image"],
                        image_sizes=[gray_image.size],
                    )
                    generated = generate_visual_margin(
                        model,
                        embeds,
                        gray_embeds,
                        tokenizer,
                        args.gen_length,
                        args.block_length,
                        args.steps,
                        args.temperature,
                        [stop_string],
                    )
                    del gray_image, gray_tensor, gray_embeds
                else:
                    generated = generate_native(
                        model,
                        embeds,
                        tokenizer,
                        args.gen_length,
                        args.block_length,
                        args.steps,
                        args.temperature,
                        [stop_string],
                    )

                row = {key: value for key, value in item.items() if key != "image"}
                row.update({"pred": parse_yes_no(generated), "raw": generated})
                rows.append(row)
                prediction_file.write(json.dumps(row) + "\n")
                prediction_file.flush()
                del image, image_tensor, input_ids, embeds, generated
                gc.collect()
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                oom += 1
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as error:
                errors += 1
                tqdm.write(f"[{item['idx']}] {type(error).__name__}: {error}")
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    output = {
        "task": "hallusion",
        "method": args.method,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "temperature": args.temperature,
        "num_predictions": len(rows),
        "oom": oom,
        "errors": errors,
        "time_s": time.time() - started,
        "metrics": score_hallusion(rows),
    }
    result_path = os.path.join(
        args.output_dir, f"hallusion_{args.method}_results.json"
    )
    with open(result_path, "w") as result_file:
        json.dump(output, result_file, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
