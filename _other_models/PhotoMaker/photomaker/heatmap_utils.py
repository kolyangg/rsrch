from __future__ import annotations
import math, types, torch, numpy as np
from typing import Dict, List, Callable

def _store(attn_maps:Dict[str,List[np.ndarray]], ln:str, a:torch.Tensor):
    # H = int(math.sqrt(a.numel()))
    # attn_maps.setdefault(ln, []).append(
    #     a.to(torch.float32).view(H, H).cpu().numpy()
    # )
    
    # if len(attn_maps.get(ln, [])) == 0:          # 1st time this layer fires
    #     mx, av = a.max().item(), a.mean().item()
    #     print(f"[HM] {ln:30s}   max={mx:6.3f}   mean={av:6.3f}")
    
    step = len(attn_maps.get(ln, []))            # how many maps so far
    mx, av = a.max().item(), a.mean().item()
    print(f"[HM] {ln:28s}  step={step:02d}  max={mx:6.3f}  mean={av:6.3f}")
    
    H = int(math.sqrt(a.numel()))
    if H * H != a.numel():
        # happens when a bogus vector slips through – just ignore it
        return
    attn_maps.setdefault(ln, []).append(
        a.to(torch.float32).view(H, H).cpu().numpy()
    )



def build_hook_identity(
    layer_name: str,
    module,
    wanted_layers: set[str],
    class_tokens_mask: torch.Tensor | None,
    num_tokens: int,
    maps_buf: dict,
    fwd_backup: dict,
   do_cfg: bool,
):
    orig_fwd = module.forward
    fwd_backup[layer_name] = orig_fwd        # so we can restore later

    def hook(hs, encoder_hidden_states=None, attention_mask=None):
        out = orig_fwd(hs, encoder_hidden_states, attention_mask)

        if layer_name not in wanted_layers or encoder_hidden_states is None:
            return out

        # • CFG → keep only conditional half
        B_all = hs.shape[0]
        if do_cfg and B_all % 2 == 0:
            hs_, enc_ = hs[B_all // 2 :], encoder_hidden_states[B_all // 2 :]
        else:
            hs_, enc_ = hs, encoder_hidden_states

        q_proj = (module.to_q if hasattr(module, "to_q") else module.q_proj)(hs_)
        k_proj = (module.to_k if hasattr(module, "to_k") else module.k_proj)(enc_)

        # --- Batch align (CFG + multi-image): tile K to Q's batch ---
        Bq = q_proj.shape[0]
        Bk = k_proj.shape[0]
        if Bk != Bq:
            rep = (Bq + Bk - 1) // Bk
            k_proj = k_proj.repeat(rep, 1, 1)[:Bq].contiguous()

        tok_idx = torch.arange(k_proj.shape[1] - num_tokens, k_proj.shape[1], device=hs.device)

        # compute logits without splitting into heads (robust to dim mismatches)
        logits = (q_proj @ k_proj.transpose(-1, -2)) * module.scale   # (B, Lq, Lk)
        sim    = logits[..., tok_idx].mean(-1)[0]                      # (Lq,)
        

        H = int(math.sqrt(sim.numel()))
        att = sim.view(H, H).to(torch.float32).detach().cpu().numpy()
        maps_buf.setdefault(layer_name, []).append(att)
        return out

    return hook

# identical to above but uses a fixed `token_idx_global` list that you
# already pass from `pipeline_NS.py`
def build_hook_focus_token(
    layer_name, module, wanted_layers,
    focus_latents, token_idx_global,
    maps_buf, fwd_backup, do_cfg,
):
    orig_fwd = module.forward
    fwd_backup[layer_name] = orig_fwd

    def hook(hs, encoder_hidden_states=None, attention_mask=None):
        out = orig_fwd(hs, encoder_hidden_states, attention_mask)
        if layer_name not in wanted_layers:
            return out

        B_all = hs.shape[0]
        if do_cfg and B_all % 2 == 0:
            hs_ = hs[B_all // 2 :]
        else:
            hs_ = hs

        # q_proj = (module.to_q if hasattr(module, "to_q") else module.q_proj)(hs_)
        # k_proj = (module.to_k if hasattr(module, "to_k") else module.k_proj)(focus_latents.to(hs_.device))

        q_proj = (module.to_q if hasattr(module, "to_q") else module.q_proj)(hs_)
        k_proj = (module.to_k if hasattr(module, "to_k") else module.k_proj)(
            focus_latents.to(hs_.device, dtype=hs_.dtype)
        )

        # --- Batch align (CFG + multi-image): tile K to Q's batch ---
        Bq = q_proj.shape[0]
        Bk = k_proj.shape[0]
        if Bk != Bq:
            rep = (Bq + Bk - 1) // Bk
            k_proj = k_proj.repeat(rep, 1, 1)[:Bq].contiguous()

        # direct logits (no head splitting)
        logits = (q_proj @ k_proj.transpose(-1, -2)) * module.scale     # (B, Lq, Lk)
        sim    = logits[..., token_idx_global].mean(-1)[0]

        H = int(math.sqrt(sim.numel()))
        maps_buf.setdefault(layer_name, []).append(
            sim.view(H, H).to(torch.float32).detach().cpu().numpy()
        )
        return out

    return hook