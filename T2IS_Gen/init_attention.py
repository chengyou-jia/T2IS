import math
import torch
import torchvision.transforms.functional as F
TOKENS = 75

def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

def init_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn" in name and "transformer_blocks" in name  and "single_transformer_blocks" not in name and module.__class__.__name__ == "Attention":
            module.forward = FluxTransformerBlock_init_forward(self, module)           
        elif "attn" in name and "single_transformer_blocks" in name and module.__class__.__name__ == "Attention":
            module.forward = FluxSingleTransformerBlock_init_forward(self, module) 

def FluxSingleTransformerBlock_init_forward(self, module):
    def forward(hidden_states=None, encoder_hidden_states=None, image_rotary_emb=None,RPG_encoder_hidden_states_list=None,RPG_norm_encoder_hidden_states_list=None,RPG_hidden_states_list=None,RPG_norm_hidden_states_list=None):
        return module.processor(module, hidden_states=hidden_states, image_rotary_emb=image_rotary_emb)
    return forward

def FluxTransformerBlock_init_forward(self, module):
    def forward(hidden_states=None, encoder_hidden_states=None, image_rotary_emb=None,RPG_encoder_hidden_states_list=None,RPG_norm_encoder_hidden_states_list=None,RPG_hidden_states_list=None,RPG_norm_hidden_states_list=None):
        return module.processor(module, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, image_rotary_emb=image_rotary_emb)
    return forward