import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=2*hidden_dim, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=2*hidden_dim, hidden_size=3*hidden_dim, num_layers=1, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(in_features=3*hidden_dim, out_features=output_dim)
        
    def forward(self, text):
        # text shape: (batch_size, max_seq_length)
        
        # Embed the input text
        embedded = text
        
        # Pass the input text through the LSTM layers
        output, (hidden, cell) = self.lstm(embedded)
        output2, (hidden2, cell2) = self.lstm2(output)
        output3, (hidden3, cell3) = self.lstm3(output2)
        
        # Concatenate the final hidden states from all the LSTM layers
        hidden_final = torch.cat((hidden[-1], hidden2[-1], hidden3[-1]), dim=-1)
        
        # Pass the concatenated hidden state through the output layer
        logits = self.fc(hidden_final)
        
        return logits