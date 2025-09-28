import torch
import torch.nn as nn

class EnhancedCharRNN(nn.Module):
    """Enhanced RNN module"""
    
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=1, rnn_type='LSTM', dropout=0.1, bidirectional=False):
        super(EnhancedCharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        else:  # RNN
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        
        # Calculate the size of the final layer input
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_output_size, output_size)
    
    def forward(self, x, lengths):
        # x: (batch_size, max_seq_len, input_size)
        # lengths: (batch_size,)
        
        batch_size = x.size(0)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), 
                                                  batch_first=True, enforce_sorted=False)
        
        # RNN forward pass
        if self.rnn_type == 'LSTM':
            packed_output, (hidden, _) = self.rnn(packed)
        else:
            packed_output, hidden = self.rnn(packed)
        
        # Get the final hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # Last layer
        
        # Apply dropout and classification layer
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output