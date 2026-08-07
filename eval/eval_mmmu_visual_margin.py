"""MMMU evaluation using fixed-quota visual-margin decoding."""
import argparse, ast, copy, gc, json, os, re, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, load_dataset

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.visual_margin import generate_visual_margin

MASK_ID, BAD_IDS = 126336, [126081, 126080, 126346, 126347]

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art","Art_Theory","Design","Music"],
    "Business": ["Accounting","Economics","Finance","Manage","Marketing"],
    "Science": ["Biology","Chemistry","Geography","Math","Physics"],
    "Health and Medicine": ["Basic_Medical_Science","Clinical_Medicine","Diagnostics_and_Laboratory_Medicine","Pharmacy","Public_Health"],
    "Humanities and Social Science": ["History","Literature","Sociology","Psychology"],
    "Tech and Engineering": ["Agriculture","Architecture_and_Engineering","Computer_Science","Electronics","Energy_and_Power","Materials","Mechanical_Engineering"],
}

def resize_image(img, max_side=384):
    w,h=img.size;m=max(w,h)
    if m<=max_side:return img
    s=max_side/m;return img.resize((int(w*s),int(h*s)),Image.LANCZOS)

def normalize_choice(text):
    return re.sub(r"[^a-z0-9.%-]+", "", str(text).lower())

def option_letter_from_text(answer_text, options):
    try:
        parsed = ast.literal_eval(options) if isinstance(options, str) else options
    except Exception:
        parsed = None
    if not isinstance(parsed, (list, tuple)):
        return None
    ans = normalize_choice(answer_text)
    if not ans:
        return None
    for i, opt in enumerate(parsed[:5]):
        opt_norm = normalize_choice(opt)
        if ans == opt_norm or ans in opt_norm or opt_norm in ans:
            return chr(ord("A") + i)
    return None

def parse_options(options):
    try:
        parsed = ast.literal_eval(options) if isinstance(options, str) else options
    except Exception:
        parsed = None
    return list(parsed) if isinstance(parsed, (list, tuple)) else None

def format_options(options):
    parsed = parse_options(options)
    if not parsed:
        return str(options)
    return "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(parsed[:5]))

def extract_answer(text, options=None):
    answer_blocks = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if answer_blocks:
        content = answer_blocks[-1].strip()
        m = re.search(r'\b([A-E])\b', content, re.IGNORECASE)
        if m: return m.group(1).upper()
        mapped = option_letter_from_text(content, options)
        if mapped: return mapped
    answer_blocks = re.findall(r'<answer>\s*(.*?)$', text, re.IGNORECASE | re.DOTALL)
    if answer_blocks:
        content = answer_blocks[-1].strip()
        m = re.search(r'\b([A-E])\b', content, re.IGNORECASE)
        if m: return m.group(1).upper()
        mapped = option_letter_from_text(content, options)
        if mapped: return mapped
    # Match: "A", "(A)", "A)", "A.", "A:", standalone letter
    # Must be word-boundary isolated, not part of a longer word
    patterns = [
        r'\b([A-E])\s*[:\)\.]',   # "A:" or "A)" or "A."
        r'[\(]([A-E])[\)]',        # "(A)"
        r'(?:answer\s*(?:is|:)?\s*)([A-E])\b',  # "answer is A" or "answer: A"
        r'\b([A-E])\b',            # standalone letter (last resort, at end of text)
    ]
    # Check first line first (most likely location)
    first_line = text.strip().split('\n')[0]
    for pat in patterns:
        m = re.findall(pat, first_line, re.IGNORECASE)
        if m: return m[0].upper()
    # Check full text
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m: return m[-1].upper()
    return '?'

@torch.no_grad()
def generate_mm(model, inputs_embeds, tokenizer, gen_length=256, block_length=256, steps=32, temperature=0.0, stopping_criteria=None):
    dev=inputs_embeds.device;mdl=model.model;lm_head=model.lm_head;et=mdl.embed_tokens
    pl=inputs_embeds.shape[1];tl=pl+gen_length
    me=et(torch.tensor([MASK_ID],device=dev))
    x_emb=me.repeat(1,tl,1);x_emb[:,:pl]=inputs_embeds
    x=torch.full((1,tl),MASK_ID,dtype=torch.long,device=dev)
    nb=gen_length//block_length;spb=steps//nb;sp=tl;fs=False
    stops=[tokenizer.encode(s,add_special_tokens=False) for s in (stopping_criteria or [])]

    for bi in range(nb):
        bs,be=pl+bi*block_length,pl+(bi+1)*block_length
        if fs and sp<=bs:break
        bmi=torch.all(torch.abs(x_emb[:,bs:be]-me)<1e-5,dim=2)
        ntt=model.get_num_transfer_tokens(bmi,spb)
        for si in range(spb):
            gc.collect();torch.cuda.empty_cache()
            mi=torch.all(torch.abs(x_emb-me)<1e-5,dim=2)
            if fs and not mi[0,pl:sp].any():break
            if not mi[0,bs:be].any():break
            with torch.cuda.amp.autocast(enabled=True):
                out=mdl(inputs_embeds=x_emb)
            hs=out.last_hidden_state;mi0=mi[0];hs_m=hs[0,mi0]
            logits_m=lm_head(hs_m).float()
            for tid in BAD_IDS:logits_m[:,tid]=-float("inf")
            x0_m=torch.argmax(logits_m,dim=-1)
            p=F.softmax(logits_m.to(torch.float32),dim=-1)
            conf_m=torch.gather(p,1,x0_m.unsqueeze(1)).squeeze(1)
            x0=x.clone();x0[0,mi0]=x0_m
            cf=torch.full((1,tl),-float("inf"),dtype=torch.float32,device=dev)
            cf[0,mi0]=conf_m;cf[:,be:]=-float("inf")
            x0=torch.where(mi,x0,x);confidence=torch.where(mi,cf,-float("inf"))
            ti=torch.zeros_like(x0,dtype=torch.bool,device=dev)
            for j in range(confidence.shape[0]):
                k=int(ntt[j,si].item())
                if k>0:
                    _,sel=torch.topk(confidence[j],k=k);ti[j,sel]=True
            x0_e=et(x0);x0_e=torch.where(mi.unsqueeze(-1).expand_as(x_emb),x0_e,x_emb)
            x_emb[ti]=x0_e[ti];x[ti]=x0[ti]
            if stopping_criteria and not fs:
                gp=x[0,pl:pl+gen_length]
                for st in stops:
                    if not isinstance(st,list):st=[st]
                    for si2 in range(gp.size(0)-len(st)+1):
                        if torch.all(gp[si2:si2+len(st)]==torch.tensor(st,device=dev)):sp=pl+si2;fs=True;break
                    if fs:break
            del hs,hs_m,logits_m,x0,cf,confidence,x0_e,out
    return tokenizer.decode(x[0,pl:sp],skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pretrained",default="GSAI-ML/LLaDA-V")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--output_dir",default="exp/mmmu_visual_margin")
    ap.add_argument("--max_samples",type=int,default=None)
    ap.add_argument("--gen_length",type=int,default=256);ap.add_argument("--block_length",type=int,default=256);ap.add_argument("--steps",type=int,default=32)
    ap.add_argument("--temperature",type=float,default=0.0)
    ap.add_argument("--dataset_arrow",default=None)
    args=ap.parse_args()
    os.makedirs(args.output_dir,exist_ok=True);device=args.device

    print(f"Loading {args.pretrained}...")
    tokenizer,model,ip,_=load_pretrained_model(args.pretrained,None,"llava_llada",attn_implementation="sdpa",device_map=device)
    model.eval()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading MMMU validation...")
    ds=Dataset.from_file(args.dataset_arrow) if args.dataset_arrow else load_dataset("lmms-lab/MMMU",split="validation")
    N=min(args.max_samples or len(ds),len(ds))
    print(f"Running {N} samples | visual margin | gen={args.gen_length} block={args.block_length} steps={args.steps}")

    cv=conv_templates["llava_llada"];stop_str=cv.sep
    results,correct,oom=[],0,0;t0=time.time()
    pred_file = open(f"{args.output_dir}/mmmu_predictions.jsonl", "w")

    for idx in tqdm(range(N),desc="MMMU"):
        r=ds[idx]
        try:
            images=[]
            for j in range(1,8):
                img=r.get(f"image_{j}")
                if img is not None:images.append(resize_image(img.convert("RGB")))
            if not images:continue
            gray_images=[Image.new("RGB",im.size,color=(128,128,128)) for im in images]
            imt=process_images(images,ip,model.config);imt=[t.to(dtype=torch.float16,device=device) for t in imt]
            gray_imt=process_images(gray_images,ip,model.config);gray_imt=[t.to(dtype=torch.float16,device=device) for t in gray_imt]
            itk=" ".join([DEFAULT_IMAGE_TOKEN]*len(images))
            prompt=(
                "This is a multiple-choice question. Use the image and the question to solve it carefully.\n"
                f"Question:\n{r['question']}\n"
                f"Choices:\n{format_options(r['options'])}\n"
                "Rules: choose exactly one listed choice. If your reasoning produces a number, phrase, or object, "
                "match it to the corresponding choice and output that choice letter only. "
                "Do not output the choice text or any unlisted value as the final answer. "
                "End the response with exactly one final tag in this format: <answer>A</answer>. /think"
            )
            conv=copy.deepcopy(cv)
            conv.append_message(conv.roles[0],itk+"\n"+prompt)
            conv.append_message(conv.roles[1],None)
            ids=tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0).to(device)
            (_,_,_,_,emb,_)=model.prepare_inputs_labels_for_multimodal(ids,None,None,None,None,imt,["image"]*len(images),image_sizes=[im.size for im in images])
            (_,_,_,_,gray_emb,_)=model.prepare_inputs_labels_for_multimodal(ids,None,None,None,None,gray_imt,["image"]*len(gray_images),image_sizes=[im.size for im in gray_images])
            if emb.shape != gray_emb.shape:
                raise RuntimeError(f"normal/gray prompt shape mismatch: {emb.shape} vs {gray_emb.shape}")
            del images,gray_images,imt,gray_imt,ids;gc.collect();torch.cuda.empty_cache()

            gen_text=generate_visual_margin(model,emb,gray_emb,tokenizer,args.gen_length,args.block_length,args.steps,args.temperature,[stop_str])
            pred=extract_answer(gen_text, r["options"]);gt=str(r["answer"]).strip().upper()
            if pred==gt:correct+=1
            sid=r["id"].split("_")[0];pat=re.compile(rf"^{sid}_(.+?)_\d+$");m=pat.search(r["id"]);sub=m.group(1) if m else "Unknown"
            detail = {"id":r["id"],"subdomain":sub,"answer":gt,"parsed_pred":pred,"correct":pred==gt,"gen_text":gen_text}
            results.append(detail)
            pred_file.write(json.dumps(detail) + "\n"); pred_file.flush()
            del emb,gray_emb,gen_text;gc.collect();torch.cuda.empty_cache()
        except torch.OutOfMemoryError:oom+=1;gc.collect();torch.cuda.empty_cache()
        except Exception as e:oom+=1;gc.collect();torch.cuda.empty_cache()

    pred_file.close()
    t=time.time()-t0
    sub2eval=defaultdict(list)
    for r_ in results:sub2eval[r_["subdomain"]].append(r_)

    printable={}
    for domain,cats in DOMAIN_CAT2SUB_CAT.items():
        in_d={}
        for cat in cats:
            if cat in sub2eval:
                s=sub2eval[cat];c2=sum(1 for x in s if x["correct"])
                in_d[cat]={"acc":c2/len(s) if s else 0,"num":len(s)}
        if in_d:
            ta=sum(r2["acc"]*r2["num"] for r2 in in_d.values());tn=sum(r2["num"] for r2 in in_d.values())
            printable[f"Overall-{domain}"]={"num":int(tn),"acc":round(ta,5)}
            for cat,r2 in in_d.items():printable[cat]={"num":int(r2["num"]),"acc":round(r2["acc"],5)}
    all_acc=correct/len(results) if results else 0
    printable["Overall"]={"num":len(results),"acc":round(all_acc,5)}

    print(f"\n=== MMMU visual margin ===")
    print(f"Valid: {len(results)-oom} OOM: {oom} | Time: {t:.0f}s")
    for k,v in sorted(printable.items()):print(f"  {k}: num={v['num']} acc={v['acc']:.4f}")
    print(f"Overall Accuracy: {all_acc:.4f} ({correct}/{len(results)})")

    with open(f"{args.output_dir}/mmmu_results.json","w") as f:
        json.dump({"overall":{"accuracy":all_acc,"correct":correct,"total":len(results),"oom":oom},"by_domain":printable,"time_s":t},f,indent=2)
    print(f"Saved to {args.output_dir}")

if __name__=="__main__":main()
