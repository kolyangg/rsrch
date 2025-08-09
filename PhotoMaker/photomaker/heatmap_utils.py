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

# def build_hook_identity(layer_name:str, module, wanted:set,
#                         class_mask, num_tokens:int,
#                         attn_maps:Dict, backup:Dict, cfg:bool):
#     """Current production recipe – selects duplicated ID tokens."""
#     orig = module.forward; backup[layer_name] = orig
#     def f(hs, encoder_hidden_states=None, attention_mask=None):
#         out = orig(hs, encoder_hidden_states, attention_mask)
#         if layer_name not in wanted or encoder_hidden_states is None:
#             return out
#         B = hs.shape[0]
#         hs_  = hs [B//2:] if cfg and B%2==0 else hs
#         enc_ = encoder_hidden_states[B//2:] if cfg and B%2==0 else encoder_hidden_states
#         q = (module.to_q if hasattr(module,"to_q") else module.q_proj)(hs_)
#         k = (module.to_k if hasattr(module,"to_k") else module.k_proj)(enc_)
#         B,L,C = q.shape; h = module.heads; d=C//h
#         Q = q.view(B,L,h,d).permute(0,2,1,3)
#         K = k.view(B,-1,h,d).permute(0,2,1,3)
#         if class_mask is not None:
#             idx = class_mask[0].to(hs.device).nonzero(as_tuple=True)[0]
#         else:
#             idx = torch.arange(K.shape[2]-num_tokens, K.shape[2], device=hs.device)
#         logit = (Q @ K.transpose(-2,-1))*module.scale
#         sim   = logit[...,idx].mean(-1).mean(1)[0]
#         _store(attn_maps, layer_name, sim)
#         return out
#     return f

# def build_hook_focus_token(layer_name:str, module,
#                            wanted:set, focus_lat:torch.Tensor,
#                            token_idx:List[int],
#                            attn_maps:Dict, backup:Dict, cfg:bool):
#     """attn_hm_NS_nosm7.py behaviour – uses “a face” K."""
#     orig = module.forward; backup[layer_name] = orig
#     def f(hs, *_args, **_kw):
#         out = orig(hs, *_args, **_kw)
#         if layer_name not in wanted: return out
#         B = hs.shape[0]
#         hs_ = hs[B//2:] if cfg and B%2==0 else hs
#         q = (module.to_q if hasattr(module,"to_q") else module.q_proj)(hs_)
#         k = (module.to_k if hasattr(module,"to_k") else module.k_proj)(
#                 focus_lat.to(hs_.dtype).repeat(hs_.shape[0],1,1))
#         B,L,C = q.shape; h = module.heads; d=C//h
#         Q = q.view(B,L,h,d).permute(0,2,1,3)
#         K = k.view(B,-1,h,d).permute(0,2,1,3)
#         logit = (Q @ K.transpose(-2,-1))*module.scale
#         if len(token_idx)>1:
#             sim = logit[...,token_idx].mean(-1).mean(1)[0]
#         else:
#             sim = logit[...,token_idx[0]].mean(1)[0]
#         _store(attn_maps, layer_name, sim)
#         return out
#     return f



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

        # B, Lq, C = q_proj.shape
        # h = module.heads
        # d = C // h

        # Q = q_proj.view(B, Lq, h, d).permute(0, 2, 1, 3)   # (B,h,Lq,d)
        # K = k_proj.view(B, -1, h, d).permute(0, 2, 1, 3)   # (B,h,Lk,d)

        # if class_tokens_mask is not None:
        #     tok_idx = class_tokens_mask[0].to(hs.device).nonzero(as_tuple=True)[0]
        # else:
        #     tok_idx = torch.arange(K.shape[2] - num_tokens, K.shape[2], device=hs.device)

        # logits = (Q @ K.transpose(-2, -1)) * module.scale          # raw logits
        # sim    = logits[..., tok_idx].mean(-1).mean(1)[0]          # (Lq,)
        
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

        q_proj = (module.to_q if hasattr(module, "to_q") else module.q_proj)(hs_)
        k_proj = (module.to_k if hasattr(module, "to_k") else module.k_proj)(focus_latents.to(hs_.device))

        # B, Lq, C = q_proj.shape
        # h = module.heads
        # d = C // h
        # Q = q_proj.view(B, Lq, h, d).permute(0, 2, 1, 3)
        # K = k_proj.view(1, -1, h, d).permute(0, 2, 1, 3)           # (1,h,Lk,d)

        # logits = (Q @ K.transpose(-2, -1)) * module.scale
        # sim    = logits[..., token_idx_global].mean(-1).mean(1)[0]
        
        # direct logits (no head splitting)
        logits = (q_proj @ k_proj.transpose(-1, -2)) * module.scale     # (B, Lq, Lk)
        sim    = logits[..., token_idx_global].mean(-1)[0]

        H = int(math.sqrt(sim.numel()))
        maps_buf.setdefault(layer_name, []).append(
            sim.view(H, H).to(torch.float32).detach().cpu().numpy()
        )
        return out

    return hook