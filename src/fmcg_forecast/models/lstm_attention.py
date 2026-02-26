"""Shared LSTM-Attention model architectures for demand and sales forecasting.

Contains:
- QuantileLoss: Pinball loss for probabilistic forecasting
- Attention: Additive attention mechanism (Bahdanau-style)
- TimeSeriesLSTMAttentionModel: BiLSTM + Attention for demand forecasting
- SeasonalFinancialLSTMModel: LSTM + MultiheadAttention for individual sales
- FinancialLSTMModel: Plain LSTM for merged sales/COGS forecasting
"""

import numpy as np
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Pinball (quantile) loss for probabilistic time series forecasting."""

    def __init__(self, quantiles: list[float]) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean quantile loss across all quantiles.

        Args:
            preds: Predictions tensor of shape (batch, horizon, num_quantiles).
            target: Ground truth tensor of shape (batch, horizon).

        Returns:
            Scalar loss tensor.
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention mechanism."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1.0 / np.sqrt(self.v.size(0)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute attention weights.

        Args:
            hidden: LSTM output tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Attention weights of shape (batch, seq_len) summing to 1.
        """
        energy = torch.tanh(self.attn(hidden))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(hidden.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention_scores, dim=1)


class TimeSeriesLSTMAttentionModel(nn.Module):
    """Bidirectional LSTM with additive attention for demand forecasting.

    Accepts continuous features + categorical embeddings for product and
    warehouse, outputs multi-step quantile forecasts.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_prob: float,
        num_quantiles: int,
        num_product_embeddings: int,
        product_embedding_dim: int,
        num_gudang_embeddings: int,
        gudang_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        lstm_input_dim = input_dim + product_embedding_dim + gudang_embedding_dim

        self.product_embedding = nn.Embedding(
            num_product_embeddings, product_embedding_dim
        )
        self.gudang_embedding = nn.Embedding(
            num_gudang_embeddings, gudang_embedding_dim
        )
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True,
        )
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim * num_quantiles)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles

    def forward(
        self,
        x_continuous: torch.Tensor,
        x_cat_product: torch.Tensor,
        x_cat_gudang: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_continuous: Continuous features (batch, seq_len, input_dim).
            x_cat_product: Product IDs (batch, 1).
            x_cat_gudang: Warehouse IDs (batch, 1).

        Returns:
            Forecast tensor of shape (batch, output_dim, num_quantiles).
        """
        embedded_prod = self.product_embedding(x_cat_product).squeeze(1)
        embedded_gudang = self.gudang_embedding(x_cat_gudang).squeeze(1)
        embedded_prod_rep = embedded_prod.unsqueeze(1).repeat(
            1, x_continuous.size(1), 1
        )
        embedded_gudang_rep = embedded_gudang.unsqueeze(1).repeat(
            1, x_continuous.size(1), 1
        )
        x_combined = torch.cat(
            [x_continuous, embedded_prod_rep, embedded_gudang_rep], dim=2
        )

        h0 = torch.zeros(
            self.num_layers * 2,
            x_combined.size(0),
            self.hidden_dim,
            device=x_combined.device,
        )
        c0 = torch.zeros(
            self.num_layers * 2,
            x_combined.size(0),
            self.hidden_dim,
            device=x_combined.device,
        )
        out, _ = self.lstm(x_combined, (h0, c0))
        attention_weights = self.attention(out)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), out).squeeze(1)
        output = self.fc(self.dropout(context_vector))
        return output.view(x_continuous.size(0), self.output_dim, self.num_quantiles)


class SeasonalFinancialLSTMModel(nn.Module):
    """LSTM + MultiheadAttention for per-product-warehouse sales forecasting.

    Concatenates sinusoidal seasonal features with the time series input.
    """

    def __init__(
        self,
        input_dim: int,
        seasonal_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim + seasonal_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout_rate, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, seasonal_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Time series input (batch, seq_len, input_dim).
            seasonal_features: Seasonal features (batch, seasonal_dim).

        Returns:
            Output tensor (batch, output_dim).
        """
        batch_size, seq_len, _ = x.shape
        seasonal_expanded = seasonal_features.unsqueeze(1).expand(-1, seq_len, -1)
        combined_input = torch.cat([x, seasonal_expanded], dim=-1)
        lstm_out, _ = self.lstm(combined_input)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = lstm_out[:, -1, :] + attn_out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class FinancialLSTMModel(nn.Module):
    """Plain LSTM for consolidated sales and COGS forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence (batch, seq_len, input_dim).

        Returns:
            Output tensor (batch, output_dim).
        """
        h0 = torch.zeros(
            self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
