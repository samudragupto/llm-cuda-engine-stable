import torch
from transformers import AutoModelForCausalLM
import struct

print("Loading TinyLlama...")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)
state = model.state_dict()

print("Exporting Draft Model (2 Layers)...")
with open("model_draft.bin", "wb") as f:
    f.write(state["model.embed_tokens.weight"].numpy().tobytes())
    
    for i in range(2):
        pref = f"model.layers.{i}."
        f.write(state[pref + "input_layernorm.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.q_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.k_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.v_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.o_proj.weight"].numpy().tobytes())
        f.write(state[pref + "post_attention_layernorm.weight"].numpy().tobytes())
        f.write(state[pref + "mlp.gate_proj.weight"].numpy().tobytes())
        f.write(state[pref + "mlp.up_proj.weight"].numpy().tobytes())
        f.write(state[pref + "mlp.down_proj.weight"].numpy().tobytes())
            
    f.write(state["model.norm.weight"].numpy().tobytes())
    f.write(state["lm_head.weight"].numpy().tobytes())

print("Draft Model Exported!")