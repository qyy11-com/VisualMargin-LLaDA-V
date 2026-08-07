"""Visual-margin reweighting for fixed-quota LLaDA-V decoding."""

import gc

import torch
import torch.nn.functional as F


MASK_ID = 126336
BAD_IDS = [126081, 126080, 126346, 126347]


@torch.no_grad()
def generate_visual_margin(
    model,
    inputs_embeds,
    gray_inputs_embeds,
    tokenizer,
    gen_length=256,
    block_length=256,
    steps=32,
    temperature=0.0,
    stopping_criteria=None,
):
    """Generate native candidates, reordering only their fixed transfer quota.

    At each step, the normal-image top-1 token remains unchanged. Its native
    confidence is multiplied by the probability that image content supports
    its margin over the normal-image runner-up. The highest-scoring positions
    are committed using exactly the same per-step quota as native decoding.
    """
    del temperature  # The aligned native baseline is deterministic.
    if inputs_embeds.shape != gray_inputs_embeds.shape:
        raise ValueError(
            "Normal and gray prompt embeddings must have identical shapes: "
            f"{tuple(inputs_embeds.shape)} != {tuple(gray_inputs_embeds.shape)}"
        )

    device = inputs_embeds.device
    backbone = model.model
    lm_head = model.lm_head
    embed_tokens = backbone.embed_tokens
    prompt_length = inputs_embeds.shape[1]
    total_length = prompt_length + gen_length
    mask_embed = embed_tokens(torch.tensor([MASK_ID], device=device))
    x_embeds = mask_embed.repeat(1, total_length, 1)
    x_embeds[:, :prompt_length] = inputs_embeds
    x = torch.full((1, total_length), MASK_ID, dtype=torch.long, device=device)

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
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
        if found_stop and stop_position <= block_start:
            break
        block_mask = torch.all(
            torch.abs(x_embeds[:, block_start:block_end] - mask_embed) < 1e-5,
            dim=2,
        )
        transfer_quota = model.get_num_transfer_tokens(block_mask, steps_per_block)

        for step_index in range(steps_per_block):
            gc.collect()
            torch.cuda.empty_cache()
            mask_index = torch.all(torch.abs(x_embeds - mask_embed) < 1e-5, dim=2)
            if found_stop and not mask_index[0, prompt_length:stop_position].any():
                break
            if not mask_index[0, block_start:block_end].any():
                break

            masked = mask_index[0]
            with torch.cuda.amp.autocast(enabled=True):
                normal_out = backbone(inputs_embeds=x_embeds)
            normal_logits = lm_head(normal_out.last_hidden_state[0, masked]).float()
            for token_id in BAD_IDS:
                normal_logits[:, token_id] = -float("inf")
            normal_top2 = torch.topk(normal_logits, k=2, dim=-1)
            candidate_ids = normal_top2.indices[:, 0]
            normal_probabilities = F.softmax(normal_logits, dim=-1)
            native_confidence = torch.gather(
                normal_probabilities, 1, candidate_ids.unsqueeze(1)
            ).squeeze(1)
            del normal_out, normal_probabilities

            gray_x_embeds = x_embeds.clone()
            gray_x_embeds[:, :prompt_length] = gray_inputs_embeds
            with torch.cuda.amp.autocast(enabled=True):
                gray_out = backbone(inputs_embeds=gray_x_embeds)
            gray_logits = lm_head(gray_out.last_hidden_state[0, masked]).float()
            gray_candidate_logits = torch.gather(gray_logits, 1, normal_top2.indices)
            normal_margin = normal_top2.values[:, 0] - normal_top2.values[:, 1]
            gray_margin = gray_candidate_logits[:, 0] - gray_candidate_logits[:, 1]
            visual_support = torch.sigmoid(normal_margin - gray_margin)
            reweighted_confidence = native_confidence * visual_support
            del gray_out, gray_logits, gray_candidate_logits, gray_x_embeds

            candidates = x.clone()
            candidates[0, masked] = candidate_ids
            score = torch.full(
                (1, total_length), -float("inf"), dtype=torch.float32, device=device
            )
            score[0, masked] = reweighted_confidence
            score[:, block_end:] = -float("inf")
            score = torch.where(mask_index, score, -float("inf"))
            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
            for batch_index in range(score.shape[0]):
                k = int(transfer_quota[batch_index, step_index].item())
                if k:
                    selected = torch.topk(score[batch_index], k=k).indices
                    transfer_index[batch_index, selected] = True

            candidate_embeds = embed_tokens(candidates)
            candidate_embeds = torch.where(
                mask_index.unsqueeze(-1).expand_as(x_embeds),
                candidate_embeds,
                x_embeds,
            )
            x_embeds[transfer_index] = candidate_embeds[transfer_index]
            x[transfer_index] = candidates[transfer_index]

            if stop_tokens and not found_stop:
                generated = x[0, prompt_length : prompt_length + gen_length]
                for stop in stop_tokens:
                    stop_tensor = torch.tensor(stop, device=device)
                    for start in range(generated.size(0) - len(stop) + 1):
                        if torch.all(generated[start : start + len(stop)] == stop_tensor):
                            stop_position = prompt_length + start
                            found_stop = True
                            break
                    if found_stop:
                        break

            del (
                normal_logits,
                normal_top2,
                candidate_ids,
                native_confidence,
                normal_margin,
                gray_margin,
                visual_support,
                reweighted_confidence,
                candidates,
                score,
                transfer_index,
                candidate_embeds,
            )

    return tokenizer.decode(
        x[0, prompt_length:stop_position], skip_special_tokens=True
    )
