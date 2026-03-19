"""
Time series forecasting model. Agent edits this file.
Baseline: simple MLP that flattens the input window and predicts the target horizon.
"""

import torch
import torch.nn as nn


class TimeSeriesMLP(nn.Module):
    """
    MLP for time series forecasting.
    Input: (batch, input_len, num_features) — flattened to (batch, input_len * num_features)
    Output: (batch, pred_len) — predicted target values
    """

    def __init__(self, input_len=96, num_features=7, pred_len=24, hidden=256):
        super().__init__()
        input_dim = input_len * num_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x):
        # x: (batch, input_len, num_features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.net(x)


def make_model(input_len=96, num_features=7, pred_len=24):
    return TimeSeriesMLP(input_len=input_len, num_features=num_features, pred_len=pred_len)
