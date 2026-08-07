"""
POPE native LLaDA-V think-mode baseline with custom memory-managed generation loop.
"""
import argparse, copy, gc, json, os, re, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

MASK_ID, BAD_IDS = 126336, [126081, 126080, 126346, 126347]

def resize_image(img, max_side=384):
    w, h = img.size; m = max(w, h)
    if m <= max_side: return img
    s = max_side / m; return img.resize((int(w*s), int(h*s)), Image.LANCZOS)

def parse_yes_no(text):
    m = re.findall(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if m: return m[-1].capitalize()
    m = re.findall(r'<answer>\s*(yes|no)', text, re.IGNORECASE)
    if m: return m[-1].capitalize()
    t = text.strip().rstrip('.')
    # Direct match
    if t.lower() in ['yes', 'no']: return t.capitalize()
    # First word is Yes/No (e.g. "No, there is no remote.")
    first_word = t.split()[0].lower().rstrip(',').rstrip('.')
    if first_word in ['yes', 'no']: return first_word.capitalize()
    # Check if output starts with Yes/No
    if t.lower().startswith('yes'): return 'Yes'
    if t.lower().startswith('no'): return 'No'
    for w in reversed(t.split()):
        w_clean = w.strip().rstrip('.,;:!').lower()
        if w_clean in ['yes', 'no']: return w_clean.capitalize()
    return 'unknown'

@torch.no_grad()
def generate_mm(model, inputs_embeds, tokenizer, gen_length=256, block_length=256, steps=32, temperature=0.0, stopping_criteria=None, return_stats=False):
    dev = inputs_embeds.device; mdl = model.model; lm_head = model.lm_head; et = mdl.embed_tokens
    pl = inputs_embeds.shape[1]; tl = pl + gen_length
    me = et(torch.tensor([MASK_ID], device=dev))
    x_emb = me.repeat(1, tl, 1); x_emb[:, :pl] = inputs_embeds
    x = torch.full((1, tl), MASK_ID, dtype=torch.long, device=dev)
    nb = gen_length // block_length; spb = steps // nb; sp = tl; fs = False
    stops = [tokenizer.encode(s, add_special_tokens=False) for s in (stopping_criteria or [])]
    stats = []

    for bi in range(nb):
        bs, be = pl + bi * block_length, pl + (bi + 1) * block_length
        if fs and sp <= bs: break
        bmi = torch.all(torch.abs(x_emb[:, bs:be] - me) < 1e-5, dim=2)
        ntt = model.get_num_transfer_tokens(bmi, spb)
        for si in range(spb):
            gc.collect(); torch.cuda.empty_cache()
            mi = torch.all(torch.abs(x_emb - me) < 1e-5, dim=2)
            if fs and not mi[0, pl:sp].any(): break
            if not mi[0, bs:be].any(): break
            with torch.cuda.amp.autocast(enabled=True):
                out = mdl(inputs_embeds=x_emb)
            hs = out.last_hidden_state; mi0 = mi[0]; hs_m = hs[0, mi0]
            logits_m = lm_head(hs_m).float()
            for tid in BAD_IDS: logits_m[:, tid] = -float("inf")
            x0_m = torch.argmax(logits_m, dim=-1)
            p = F.softmax(logits_m.to(torch.float32), dim=-1)
            conf_m = torch.gather(p, 1, x0_m.unsqueeze(1)).squeeze(1)
            x0 = x.clone(); x0[0, mi0] = x0_m
            cf = torch.full((1, tl), -float("inf"), dtype=torch.float32, device=dev)
            cf[0, mi0] = conf_m; cf[:, be:] = -float("inf")
            x0 = torch.where(mi, x0, x); confidence = torch.where(mi, cf, -float("inf"))
            ti = torch.zeros_like(x0, dtype=torch.bool, device=dev)
            for j in range(confidence.shape[0]):
                k = int(ntt[j, si].item())
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k); ti[j, sel] = True
            if return_stats:
                selected_token_ids = x0[0, sel]
                stats.append({
                    "block": bi,
                    "step": si,
                    "quota": int(ntt[0, si].item()),
                    "transferred": int(ti.sum().item()),
                    "remaining": int(mi[0, bs:be].sum().item() - ti.sum().item()),
                    "selection_order_positions": (sel - pl).tolist(),
                    "selection_order_token_ids": selected_token_ids.tolist(),
                    "selection_order_token_text": [
                        tokenizer.decode([int(token_id)], skip_special_tokens=False)
                        for token_id in selected_token_ids.tolist()
                    ],
                    "selection_order_native_confidence": confidence[0, sel].tolist(),
                })
            x0_e = et(x0); x0_e = torch.where(mi.unsqueeze(-1).expand_as(x_emb), x0_e, x_emb)
            x_emb[ti] = x0_e[ti]; x[ti] = x0[ti]
            if stopping_criteria and not fs:
                gp = x[0, pl:pl+gen_length]
                for st in stops:
                    if not isinstance(st, list): st = [st]
                    for si2 in range(gp.size(0) - len(st) + 1):
                        if torch.all(gp[si2:si2+len(st)] == torch.tensor(st, device=dev)): sp = pl + si2; fs = True; break
                    if fs: break
            del hs, hs_m, logits_m, x0, cf, confidence, x0_e, out
    text = tokenizer.decode(x[0, pl:sp], skip_special_tokens=True)
    return (text, stats) if return_stats else text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="GSAI-ML/LLaDA-V")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_dir", default="exp/pope_native")
    ap.add_argument(
        "--pope_root",
        default=os.environ.get("POPE_ROOT"),
        help="POPE root containing output/coco and val2014/val2014",
    )
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--split", choices=["random", "popular", "adversarial"], default="random")
    ap.add_argument("--sample_indices", default=None)
    ap.add_argument("--save_decode_stats", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gen_length", type=int, default=256); ap.add_argument("--block_length", type=int, default=256); ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True); device = args.device

    print(f"Loading {args.pretrained}...")
    tokenizer, model, ip, _ = load_pretrained_model(args.pretrained, None, "llava_llada", attn_implementation="sdpa", device_map=device)
    model.eval()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not args.pope_root:
        ap.error("--pope_root or POPE_ROOT is required")
    data_path = os.path.join(
        args.pope_root, "output", "coco", f"coco_pope_{args.split}.json"
    )
    with open(data_path) as f:
        data = [json.loads(l) for l in f]
    items = []
    img_base = os.path.join(args.pope_root, "val2014", "val2014")
    for item in data:
        p = os.path.join(img_base, item["image"])
        if os.path.exists(p): items.append({"path": p, "query": item["text"], "label": item["label"].capitalize(), "id": item["question_id"]})
    if args.sample_indices:
        run_indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    else:
        N = min(args.max_samples or len(items), len(items))
        run_indices = list(range(N))
    cv = conv_templates["llava_llada"]; stop_str = cv.sep
    prediction_path = f"{args.output_dir}/pope_predictions.jsonl"
    existing_rows = []
    if args.resume and os.path.exists(prediction_path):
        with open(prediction_path) as existing_file:
            for line in existing_file:
                try:
                    existing_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        completed_indices = {row["idx"] for row in existing_rows}
        run_indices = [idx for idx in run_indices if idx not in completed_indices]
    predictions = [row["pred"] for row in existing_rows]
    labels = [row["label"] for row in existing_rows]
    correct = sum(bool(row["correct"]) for row in existing_rows)
    oom = 0
    print(
        f"Running {len(run_indices)} remaining samples"
        f" ({len(existing_rows)} resumed) | split={args.split} native think | "
        f"gen={args.gen_length} block={args.block_length} steps={args.steps}"
    )
    t0 = time.time()

    pred_file = open(prediction_path, "a" if args.resume else "w")

    for idx in tqdm(run_indices, desc="POPE"):
        it = items[idx]
        try:
            img = resize_image(Image.open(it["path"]).convert("RGB"))
            imt = process_images([img], ip, model.config); imt = [t.to(dtype=torch.float16, device=device) for t in imt]
            prompt = (
                "This is a binary visual existence question. Use the image to decide whether the object asked about is present.\n"
                f"Question: {it['query']}\n"
                "Rules: answer yes only if the asked object is visible or clearly implied by the image; "
                "answer no if it is absent, uncertain, or only a different object is visible. "
                "The final answer must be exactly yes or no, not a sentence. "
                "End the response with exactly one final tag in this format: <answer>yes</answer> or <answer>no</answer>. /think"
            )
            conv = copy.deepcopy(cv)
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, imt, ["image"], image_sizes=[img.size])
            del img, imt, ids; gc.collect(); torch.cuda.empty_cache()

            generated = generate_mm(
                model, emb, tokenizer, args.gen_length, args.block_length,
                args.steps, args.temperature, [stop_str],
                return_stats=args.save_decode_stats)
            if args.save_decode_stats:
                gen_text, decode_stats = generated
            else:
                gen_text, decode_stats = generated, None
            pred = parse_yes_no(gen_text); label = it["label"]
            predictions.append(pred); labels.append(label)
            ok = "OK" if pred == label else "WRONG"
            if pred == label: correct += 1
            detail = {"idx": idx, "query": it["query"], "label": label, "pred": pred, "correct": pred==label, "raw": gen_text}
            if decode_stats is not None:
                detail["decode_stats"] = decode_stats
            pred_file.write(json.dumps(detail) + "\n"); pred_file.flush()
            if idx < 5:
                tqdm.write(f"[{idx}] Q: {it['query'][:60]} | label={label} pred={pred} [{ok}]")
                tqdm.write(f"  {gen_text[:150]}")
            del emb, gen_text; gc.collect(); torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            oom += 1; tqdm.write(f"[{idx}] OOM"); gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            oom += 1; tqdm.write(f"[{idx}] Error: {str(e)[:80]}"); gc.collect(); torch.cuda.empty_cache()
    pred_file.close()

    tp = tn = fp = fn = 0
    for p, l in zip(predictions, labels):
        if p == "Yes" and l == "Yes": tp += 1
        elif p == "Yes" and l == "No": fp += 1
        elif p == "No" and l == "No": tn += 1
        elif p == "No" and l == "Yes": fn += 1
    total = tp + tn + fp + fn
    acc = (tp+tn)/total if total else 0; prec = tp/(tp+fp) if (tp+fp) else 0; rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    t = time.time()-t0

    print(f"\n=== POPE native think ===")
    print(f"Valid: {total} OOM: {oom} | Time: {t:.0f}s ({total/t:.3f} samples/s)")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    with open(f"{args.output_dir}/pope_results.json", "w") as f:
        json.dump({"split": args.split, "total": total, "resumed": len(existing_rows), "oom": oom, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "time_s": t}, f, indent=2)
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__": main()
