import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The original model uses nn.LSTM which is already highly optimized via cuDNN.
# However, the forward pass performs a slicing operation `out[:, -1, :]` and then a Linear layer.
# In the provided Model.forward, the return value is actually `state[0]`, which is the 
# hidden state of the last layer (h_n).
#
# To optimize, we can fuse the extraction of the last hidden state and the linear layer 
# if the goal was the FC output, but since the model returns `state[0]`, 
# the FC layer is actually a dead computation in the original code's return path.
#
# Looking closely at the original code:
# out, state = self.lstm(x, (h0, c0))
# out = self.fc(out[:, -1, :])
# return state[0]
#
# The variable 'out' (the result of the FC layer) is never used. The function returns 'state[0]'.
# Therefore, the most significant optimization is to remove the unused FC computation.
#
# If we assume the user intended to return the FC output, we would fuse the last-step 
# extraction and the linear layer. But to stay faithful to the provided logic:
# We will implement a custom kernel that handles the LSTM and returns the state[0] 
# directly, bypassing the unnecessary FC calculation.

# Since nn.LSTM is a black box calling cuDNN, we cannot easily "fuse" into it 
# without rewriting the entire cuDNN LSTM kernel (which is massive).
# Instead, we optimize the execution flow.

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # We keep the LSTM as it is the most compute-intensive part and uses cuDNN.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=False)
        
        # We keep the FC layer in case the user changes the return statement,
        # but we recognize it's currently a no-op for the return value.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        Optimized forward pass.
        The original code calculates:
            out, state = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return state[0]
        
        Note that 'out' is computed but not returned. 
        The return value is 'state[0]', which is the hidden state of the last layer.
        """
        
        # Perform LSTM computation
        # state is a tuple (h_n, c_n)
        # h_n shape: (num_layers, batch, hidden_size)
        _, state = self.lstm(x, (h0, c0))
        
        # The original code performs:
        # out = self.fc(out[:, -1, :])
        # but then returns state[0].
        # To be mathematically identical to the provided code, we return state[0].
        # This avoids the overhead of the FC layer and the slicing of 'out'.
        
        return state[0]

# If the user actually intended to return the FC output (which is common in such architectures),
# the optimized version would be:
class ModelNewCorrected(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNewCorrected, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        # We use the fact that the last hidden state of the last layer 
        # is equivalent to the last time step of the output for a non-bidirectional LSTM.
        # out[:, -1, :] == state[0][-1, :, :]
        out, state = self.lstm(x, (h0, c0))
        # Instead of slicing the large 'out' tensor, we slice the 'state' tensor 
        # which is much smaller (num_layers vs seq_len).
        last_h = state[0][-1, :, :] 
        return self.fc(last_h)

# However, following the prompt's requirement to optimize the *given* architecture:
# The given architecture returns state[0].