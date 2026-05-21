import torch
from transformers import AutoModelForCausalLM, AutoConfig
from torch.utils.cpp_extension import load_inline

# The Reformer architecture uses LSH (Locality Sensitive Hashing) attention.
# A common bottleneck in transformer-like models is the final Logits computation:
# logits = hidden_states @ embedding_weight.T
# We can optimize this by implementing a fused kernel that handles the 
# projection and potentially applies scaling or bias if needed, 
# but specifically for large batch sizes and small sequence lengths, 
# a custom GEMM-like kernel or a fused projection can be beneficial.
# However, since the model is a black box (AutoModelForCausalLM), 
# the most effective way to optimize the "Model" wrapper without 
# rewriting the entire Reformer implementation is to optimize the 
# heavy-duty operations if we can intercept them.
# Given the constraints, we will implement a fused kernel for the 
# final linear projection (Logits) which is often the bottleneck 
# in large-vocabulary causal models.

fused_logits_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel for: logits = hidden_states @ weight.T
// This is a simplified version of a GEMM kernel optimized for the 
// specific case of projecting hidden states to vocab size.
__global__ void fused_logits_kernel(const float* __restrict__ hidden_states, 
                                    const float* __restrict__ weight, 
                                    float* __restrict__ logits, 
                                    int batch_seq, int hidden_dim, int vocab_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch * seq
    int col = blockIdx.x * blockDim.x + threadIdx.x; // vocab_size

    if (row < batch_seq && col < vocab_size) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; ++k) {
            sum += hidden_states[row * hidden_dim + k] * weight[col * hidden_dim + k];
        }
        logits[row * vocab_size + col] = sum;
    }
}

torch::Tensor fused_logits_cuda(torch::Tensor hidden_states, torch::Tensor weight) {
    const int batch_seq = hidden_states.size(0) * hidden_states.size(1);
    const int hidden_dim = hidden_states.size(2);
    const int vocab_size = weight.size(0);

    auto logits = torch::empty({hidden_states.size(0), hidden_states.size(1), vocab_size}, hidden_states.options());

    dim3 block_size(16, 16);
    dim3 grid_size((vocab_size + block_size.x - 1) / block_size.x, 
                   (batch_seq + block_size.y - 1) / block_size.y);

    fused_logits_kernel<<<grid_size, block_size>>>(
        hidden_states.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        logits.data_ptr<float>(), 
        batch_seq, hidden_dim, vocab_size
    );

    return logits;
}
"""

fused_logits_cpp_source = "torch::Tensor fused_logits_cuda(torch::Tensor hidden_states, torch::Tensor weight);"

fused_ops = load_inline(
    name="fused_logits_ops",
    cpp_sources=fused_logits_cpp_source,
    cuda_sources=fused_logits_source,
    functions=["fused_logits_cuda"],
    verbose=False,
)

class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.fused_ops = fused_ops
        
        # We attempt to hook into the LM head to use our optimized kernel.
        # In Transformers, the logits are usually computed via self.lm_head(hidden_states).
        # We replace the lm_head forward pass logic.
        self.lm_head = self.model.lm_head
        self.weight = self.lm_head.weight

    def forward(self, x):
        # Get the hidden states from the transformer backbone
        outputs = self.model.transformer(x) if hasattr(self.model, 'transformer') else self.model.model(x)
        
        # If the model returns a tuple (common in HF), get the last element
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # If hidden_states is a BaseModelOutput, extract last_hidden_state
        if hasattr(hidden_states, 'last_hidden_state'):
            hidden_states = hidden_states.last_hidden_state

        # Use the custom fused kernel for the final projection to logits
        # This replaces: logits = self.lm_head(hidden_states)
        # Note: We assume FP32 as per instructions.
        logits = self.fused_ops.fused_logits_cuda(hidden_states.contiguous(), self.weight.contiguous())
        
        return logits