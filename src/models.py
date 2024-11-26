import torch.nn as nn

class LinearRegression_Model(nn.Module):
    """
    
    """
    def __init__(self, input_size, lstm_hidden_size, rnn_hidden_size, num_classes):
        super(LinearRegression_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        self.rnn = nn.RNN(lstm_hidden_size, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
        self.drop = nn.Dropout(p=0.1)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        rnn_out, _ = self.rnn(lstm_out)
        out = self.drop(self.fc(rnn_out[:, -1, :]))
        return out