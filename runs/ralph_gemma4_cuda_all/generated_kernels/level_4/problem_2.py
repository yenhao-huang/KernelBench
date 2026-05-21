import torch
from transformers import AutoConfig, AutoModelForCausalLM

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch.dtype=torch.float32)

    def forward(self, x):
        return self.model(x).logits

def get_inputs():
    # ...