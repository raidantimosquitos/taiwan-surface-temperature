import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based model for time series prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM outputs: (hidden states, (h_n, c_n))
        lstm_out, _ = self.lstm(x)
        # Use the last time step's hidden state
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out


class GRUModel(nn.Module):
    """
    GRU-based model for time series prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out