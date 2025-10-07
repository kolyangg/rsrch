# +++ pulid/attention_processor_NS2.py

import torch, torch.nn as nn
from typing import Optional

# Reuse your PM branched processors
from photomaker.attn_processor import (
    BranchedAttnProcessor,
    BranchedCrossAttnProcessor,
)
# Reuse PuLID ID processor without touching it
from pulid.attention_processor import IDAttnProcessor
from diffusers.models.attention_processor import AttnProcessor2_0

class IDBranchedCrossAttnProcessor(BranchedCrossAttnProcessor):
    """
    Cross-attn that preserves PuLID’s ID injection while applying your
    face/background branching & mask mixing.
    Expects `cross_attention_kwargs` to optionally carry:
       - 'id_embedding': Tensor[B, Lid, C] from PuLID (already computed)
       - 'id_scale'    : float
    """
    # Tell diffusers we accept cross_attention_kwargs (prevents warnings)
    supports_cross_attention_kwargs = True
    
    def __init__(self, hidden_size:int, cross_attention_dim:int, scale:float=1.0, num_tokens:int=77):
        super().__init__(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                         scale=scale, num_tokens=num_tokens)
        self._id_proc = IDAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    # def __call__(self, attn, hidden_states, encoder_hidden_states=None,
    #              attention_mask=None, temb=None, **kw):
    #     # 1) Run *branched* cross-attn (this handles mask mix + optional id_embeds routed via .id_embeds)
    #     out = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)
    #     # 2) Add PuLID ID-adapter residual on top (exact same math as their processor)

    # def __call__(self, attn, hidden_states, encoder_hidden_states=None,
    #              attention_mask=None, temb=None, **kw):
    #     # Ensure id_embedding dtype/device matches UNet weights
    #     ca_kwargs = (kw.get("cross_attention_kwargs") or {}) if "cross_attention_kwargs" in kw else {}
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, cross_attention_kwargs=None, **kw):
       # Ensure id_embedding dtype/device matches UNet weights
        ca_kwargs = cross_attention_kwargs or {}
    
        if isinstance(ca_kwargs, dict):
            id_emb = ca_kwargs.get("id_embedding", None)
            if id_emb is not None:
                w_dtype = attn.to_k.weight.dtype
                id_emb = id_emb.to(device=hidden_states.device, dtype=w_dtype)
                ca_kwargs = dict(ca_kwargs, id_embedding=id_emb)
                kw["cross_attention_kwargs"] = ca_kwargs
                cross_attention_kwargs = ca_kwargs

        # If no masks set yet, fall back to pure PuLID ID processor (vanilla path)
        if getattr(self, "mask", None) is None and getattr(self, "mask_ref", None) is None:
            return self._id_proc(attn, hidden_states, encoder_hidden_states,
                                 attention_mask, temb, cross_attention_kwargs=cross_attention_kwargs, **kw)

        # # 1) Run *branched* cross-attn (uses masks set by two_branch_predict)
        # out = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)
        

        # 1) Run *branched* cross-attn (uses masks set by two_branch_predict)
        # out = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb,
        #                        cross_attention_kwargs=cross_attention_kwargs, **kw)
        out = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                               cross_attention_kwargs=ca_kwargs, **kw)
        
        # 2) Add PuLID ID-adapter residual on top (same math as their processor)

        # id_embedding = (kw.get("cross_attention_kwargs", {}) or {}).get("id_embedding", None)
        # id_scale     = float((kw.get("cross_attention_kwargs", {}) or {}).get("id_scale", 1.0))

        id_embedding = ca_kwargs.get("id_embedding", None)
        id_scale     = float(ca_kwargs.get("id_scale", 1.0))

        if id_embedding is not None and id_scale != 0.0:
            out = out + id_scale * (
                self._id_proc(attn, hidden_states, encoder_hidden_states,
                              attention_mask, temb, id_embedding=id_embedding,
                            #   id_scale=1.0, cross_attention_kwargs=cross_attention_kwargs) - hidden_states
                              id_scale=1.0, cross_attention_kwargs=ca_kwargs) - hidden_states
            )
        return out


    # allow two_branch_predict to update masks each step
    def set_masks(self, mask, mask_ref=None):
        super().set_masks(mask, mask_ref)


class SoftBranchedAttnProcessor(nn.Module):
    # Accept cross_attention_kwargs to silence warnings on attn1
    supports_cross_attention_kwargs = True
    """
    Self-attn that uses BranchedAttnProcessor when masks are set,
    otherwise falls back to vanilla AttnProcessor2_0 (no-branch).
    """
    def __init__(self, hidden_size:int, scale:float=1.0):
        super().__init__()
        self._branched = BranchedAttnProcessor(hidden_size=hidden_size,
                                               cross_attention_dim=hidden_size,
                                               scale=scale)
        self._vanilla = AttnProcessor2_0()
        self.mask = None
        self.mask_ref = None

    def set_masks(self, mask, mask_ref=None):
        self.mask = mask
        self.mask_ref = mask_ref if mask_ref is not None else mask
        if hasattr(self._branched, "set_masks"):
            self._branched.set_masks(self.mask, self.mask_ref)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, cross_attention_kwargs=None, **kw):

        # choose branched only if masks are available
        if self.mask is None and self.mask_ref is None:
        #     return self._vanilla(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)
        # return self._branched(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)

            return self._vanilla(attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                                 cross_attention_kwargs=cross_attention_kwargs, **kw)

        return self._branched(attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                              cross_attention_kwargs=cross_attention_kwargs, **kw)



def make_pulid_branched_processors(unet, cross_attention_dim:int, num_tokens:int=77, scale:float=1.0):
    """
    Build a dict[name->processor] for UNet identical to PM patcher:
      - attn1 -> BranchedAttnProcessor
      - attn2 -> IDBranchedCrossAttnProcessor
    """
    procs = {}
    for name in unet.attn_processors.keys():
        # hidden size per block like PM code
        if "mid_block" in name:
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            b = int(name[len("up_blocks."):].split(".")[0])
            hidden = list(reversed(unet.config.block_out_channels))[b]
        elif name.startswith("down_blocks"):
            b = int(name[len("down_blocks."):].split(".")[0])
            hidden = unet.config.block_out_channels[b]
        else:
            hidden = unet.config.block_out_channels[0]

        # if name.endswith("attn1.processor"):
        #     procs[name] = SoftBranchedAttnProcessor(hidden_size=hidden,
        #                                         scale=scale).to(unet.device, dtype=unet.dtype)
        if name.endswith("attn1.processor"):
            proc = SoftBranchedAttnProcessor(hidden_size=hidden, scale=scale)
            if hasattr(proc, "to"):
                proc = proc.to(unet.device, dtype=unet.dtype)
            procs[name] = proc
        
        elif name.endswith("attn2.processor"):
            procs[name] = IDBranchedCrossAttnProcessor(hidden_size=hidden,
                                                       cross_attention_dim=cross_attention_dim,
                                                       scale=scale, num_tokens=num_tokens
                                                      ).to(unet.device, dtype=unet.dtype)
    return procs
