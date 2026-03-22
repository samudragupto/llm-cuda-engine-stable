import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import struct

def quantize_to_int8(tensor):
    # Per-row quantization to match your C++ read_qt
    # tensor shape: [out_features, in_features] (PyTorch default)
    # C++ expects: Row-major
    
    # Calculate scale per row: max(abs(row)) / 127
    scales = tensor.abs().max(dim=1).values / 127.0
    scales = scales.to(torch.float16)
    
    # Quantize: x / scale
    # We broaden dimensions for broadcast
    quantized = (tensor / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    
    return quantized.numpy(), scales.numpy()

print("Loading TinyLlama...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
state = model.state_dict()

# Configuration matches TinyLlama
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 5632
NUM_LAYERS = 22  # Export ALL layers
HEADS = 32
KV_HEADS = 4

print(f"Exporting Mixed Precision Model ({NUM_LAYERS} layers)...")
with open("model_mixed.bin", "wb") as f:
    # 1. Token Embeddings (FP16)
    f.write(state["model.embed_tokens.weight"].numpy().tobytes())
    
    # 2. Layers
    for i in range(NUM_LAYERS):
        print(f"Processing Layer {i+1}/{NUM_LAYERS}...")
        pref = f"model.layers.{i}."
        
        # Attention Norm (FP16)
        f.write(state[pref + "input_layernorm.weight"].numpy().tobytes())
        
        # Attention Projections (FP16)
        f.write(state[pref + "self_attn.q_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.k_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.v_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.o_proj.weight"].numpy().tobytes())
        
        # FFN Norm (FP16)
        f.write(state[pref + "post_attention_layernorm.weight"].numpy().tobytes())
        
        # MLP Weights (INT8 + Scales)
        # W1 (Gate), W2 (Up), W3 (Down)
        # Note: In TinyLlama:
        # gate_proj -> W1
        # up_proj   -> W2
        # down_proj -> W3
        
        q_gate, s_gate = quantize_to_int8(state[pref + "mlp.gate_proj.weight"])
        f.write(q_gate.tobytes())
        f.write(s_gate.tobytes())
        
        q_up, s_up = quantize_to_int8(state[pref + "mlp.up_proj.weight"])
        f.write(q_up.tobytes())
        f.write(s_up.tobytes())
        
        q_down, s_down = quantize_to_int8(state[pref + "mlp.down_proj.weight"])
        f.write(q_down.tobytes())
        f.write(s_down.tobytes())
            
    # 3. Final Norm (FP16)
    f.write(state["model.norm.weight"].numpy().tobytes())
    
    # 4. LM Head (FP16)
    f.write(state["lm_head.weight"].numpy().tobytes())

print("Model Exported!")

# 5. Export Tokenizer Vocabulary
print("Exporting Tokenizer...")
vocab = tokenizer.get_vocab()
# Sort by ID to ensure index matches
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
max_id = sorted_vocab[-1][1]

with open("tokenizer.bin", "wb") as f:
    # Write vocab size
    f.write(struct.pack("I", max_id + 1))
    
    # Write tokens in order
    for i in range(max_id + 1):
        token_str = tokenizer.convert_ids_to_tokens(i)
        if token_str is None:
            token_str = ""
        # Handle byte conversion for special chars
        encoded_token = token_str.encode('utf-8', errors='replace')
        f.write(struct.pack("I", len(encoded_token)))
        f.write(encoded_token)

print("Tokenizer Exported!")