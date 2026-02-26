import pytest
import torch


@pytest.mark.unit
def test_quantile_loss_forward():
    from fmcg_forecast.models.lstm_attention import QuantileLoss

    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    preds = torch.randn(4, 10, 3)  # batch=4, horizon=10, quantiles=3
    target = torch.randn(4, 10)
    loss = loss_fn(preds, target)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


@pytest.mark.unit
def test_attention_module():
    from fmcg_forecast.models.lstm_attention import Attention

    attn = Attention(hidden_dim=32)
    hidden = torch.randn(4, 21, 32)  # batch=4, seq=21, hidden=32
    weights = attn(hidden)
    assert weights.shape == (4, 21)
    # Weights should sum to ~1 per batch
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


@pytest.mark.unit
def test_lstm_attention_model_forward():
    from fmcg_forecast.models.lstm_attention import TimeSeriesLSTMAttentionModel

    model = TimeSeriesLSTMAttentionModel(
        input_dim=12,
        hidden_dim=24,
        num_layers=2,
        output_dim=60,
        dropout_prob=0.25,
        num_quantiles=1,
        num_product_embeddings=10,
        product_embedding_dim=16,
        num_gudang_embeddings=5,
        gudang_embedding_dim=8,
    )
    x_cont = torch.randn(4, 21, 12)
    x_cat_prod = torch.randint(0, 10, (4, 1))
    x_cat_gudang = torch.randint(0, 5, (4, 1))
    output = model(x_cont, x_cat_prod, x_cat_gudang)
    assert output.shape == (4, 60, 1)


@pytest.mark.unit
def test_sales_lstm_model_forward():
    from fmcg_forecast.models.lstm_attention import SeasonalFinancialLSTMModel

    model = SeasonalFinancialLSTMModel(
        input_dim=1,
        seasonal_dim=6,
        hidden_dim=72,
        num_layers=3,
        output_dim=1,
        dropout_rate=0.2,
    )
    x = torch.randn(4, 30, 1)
    seasonal = torch.randn(4, 6)
    output = model(x, seasonal)
    assert output.shape == (4, 1)


@pytest.mark.unit
def test_merged_lstm_model_forward():
    from fmcg_forecast.models.lstm_attention import FinancialLSTMModel

    model = FinancialLSTMModel(
        input_dim=8,
        hidden_dim=72,
        num_layers=3,
        output_dim=1,
    )
    x = torch.randn(4, 30, 8)
    output = model(x)
    assert output.shape == (4, 1)
