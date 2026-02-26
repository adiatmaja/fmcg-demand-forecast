"""Individual sales forecaster — per region-warehouse LSTM with seasonal attention.

Rewrites yokulak-forecasting/sales/individual/forecasting.py without
any MySQL/database dependencies.
"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fmcg_forecast.config import SalesConfig
from fmcg_forecast.models.lstm_attention import SeasonalFinancialLSTMModel


logger = logging.getLogger(__name__)


def preprocess_raw_data(
    df: pd.DataFrame,
    holiday_dates: set,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Adjust dates for non-business days and aggregate into daily sales/COGS.

    Args:
        df: Raw sales DataFrame with date, id_region, id_gudang, sales, cogs.
        holiday_dates: Set of date objects representing public holidays.
        date_col: Name of the date column.

    Returns:
        Dict with 'sales' and 'cogs' DataFrames aggregated by date/region/gudang.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    def _next_valid(d: pd.Timestamp) -> pd.Timestamp:
        while d.weekday() == 6 or d.date() in holiday_dates:
            d += timedelta(days=1)
        return d

    df[date_col] = df[date_col].apply(_next_valid)
    df_daily = (
        df.groupby([date_col, "id_gudang", "id_region"])
        .agg({"sales": "sum", "cogs": "sum"})
        .reset_index()
    )
    logger.info("Preprocessed %d daily records", len(df_daily))

    sales_df = df_daily[[date_col, "id_region", "id_gudang", "sales"]].copy()
    cogs_df = df_daily[[date_col, "id_region", "id_gudang", "cogs"]].copy()
    return {"sales": sales_df, "cogs": cogs_df}


class SeasonalFinancialForecaster:
    """LSTM + MultiheadAttention forecaster for per-region-warehouse sales/COGS.

    Trains one model per (metric, region, gudang) pair.
    """

    SEASONAL_DIM = 6

    def __init__(self, config: SalesConfig, holiday_dates: set) -> None:
        self.config = config
        self.holidays = holiday_dates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers: dict[str, MinMaxScaler] = {}
        self.seasonal_scalers: dict[str, StandardScaler] = {}
        self.models: dict[str, SeasonalFinancialLSTMModel] = {}
        logger.info("SeasonalFinancialForecaster using device: %s", self.device)

    def is_valid_business_day(self, d: pd.Timestamp) -> bool:
        """Return True if d is not a Sunday and not a holiday."""
        return d.weekday() != 6 and d.date() not in self.holidays

    def create_seasonal_features(self, dates: list[pd.Timestamp]) -> np.ndarray:
        """Create 6-dimensional sinusoidal seasonal features.

        Args:
            dates: List of timestamps.

        Returns:
            Array of shape (len(dates), 6).
        """
        features = []
        for d in dates:
            week_num = d.isocalendar()[1]
            features.append(
                [
                    np.sin(2 * np.pi * d.weekday() / 7),
                    np.cos(2 * np.pi * d.weekday() / 7),
                    np.sin(2 * np.pi * d.month / 12),
                    np.cos(2 * np.pi * d.month / 12),
                    np.sin(2 * np.pi * week_num / 52),
                    np.cos(2 * np.pi * week_num / 52),
                ]
            )
        return np.array(features, dtype=np.float32)

    def prepare_data_with_seasonality(
        self,
        df: pd.DataFrame,
        metric: str,
        id_region: str,
        id_gudang: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, pd.DataFrame | None]:
        """Prepare scaled sequences + seasonal features for one region-gudang pair.

        Args:
            df: DataFrame with date, id_region, id_gudang, and metric columns.
            metric: Column name ('sales' or 'cogs').
            id_region: Region identifier.
            id_gudang: Warehouse identifier.

        Returns:
            Tuple of (X, y, seasonal_features, processed_df) or (None,)*4.
        """
        filtered = (
            df[(df["id_region"] == id_region) & (df["id_gudang"] == id_gudang)]
            .copy()
            .sort_values("date")
        )
        min_len = self.config.input_window + 1
        if len(filtered) < min_len:
            logger.warning(
                "Insufficient data for %s/%s: need %d, got %d",
                id_region, id_gudang, min_len, len(filtered),
            )
            return None, None, None, None

        filtered[metric] = filtered[metric].ffill().fillna(0)
        key = f"{metric}_r{id_region}_g{id_gudang}"
        self.scalers[key] = MinMaxScaler()
        scaled = self.scalers[key].fit_transform(filtered[[metric]].values)

        dates_list = filtered["date"].tolist()
        seasonal_raw = self.create_seasonal_features(dates_list)
        s_key = f"{key}_seasonal"
        self.seasonal_scalers[s_key] = StandardScaler()
        scaled_seasonal = self.seasonal_scalers[s_key].fit_transform(seasonal_raw)

        X_list, y_list, s_list = [], [], []
        for i in range(len(scaled) - self.config.input_window):
            X_list.append(scaled[i : i + self.config.input_window])
            y_list.append(scaled[i + self.config.input_window, 0])
            s_list.append(scaled_seasonal[i + self.config.input_window])

        X = torch.FloatTensor(np.array(X_list)).to(self.device)
        y = torch.FloatTensor(np.array(y_list).reshape(-1, 1)).to(self.device)
        seasonal = torch.FloatTensor(np.array(s_list)).to(self.device)
        return X, y, seasonal, filtered

    def train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        seasonal: torch.Tensor,
        model_key: str,
    ) -> SeasonalFinancialLSTMModel | None:
        """Train a SeasonalFinancialLSTMModel for one metric/region/gudang.

        Args:
            X: Input sequences tensor.
            y: Target values tensor.
            seasonal: Seasonal feature tensor (aligned to targets).
            model_key: Unique key for storing the trained model.

        Returns:
            Trained model or None if data is insufficient.
        """
        if X is None or len(X) == 0:
            logger.error("No data to train model for %s", model_key)
            return None

        split = int(len(X) * 0.8)
        if split == 0:
            X_tr, X_val = X, X
            y_tr, y_val = y, y
            s_tr, s_val = seasonal, seasonal
        else:
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]
            s_tr, s_val = seasonal[:split], seasonal[split:]

        model = SeasonalFinancialLSTMModel(
            input_dim=1,
            seasonal_dim=self.SEASONAL_DIM,
            hidden_dim=self.config.model_params.hidden_dim,
            num_layers=self.config.model_params.num_layers,
            output_dim=1,
            dropout_rate=self.config.model_params.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.model_params.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=self.config.patience // 2, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.epochs):
            model.train()
            perm = torch.randperm(X_tr.size(0))
            for i in range(0, X_tr.size(0), self.config.batch_size):
                idx = perm[i : i + self.config.batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_tr[idx], s_tr[idx]), y_tr[idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val, s_val), y_val).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.debug("Early stop at epoch %d for %s", epoch + 1, model_key)
                break

        if best_state:
            model.load_state_dict(best_state)
        self.models[model_key] = model
        return model

    def generate_seasonal_forecast(
        self,
        df_historical: pd.DataFrame,
        metric: str,
        id_region: str,
        id_gudang: str,
    ) -> pd.DataFrame | None:
        """Generate autoregressive multi-step forecast for one region-gudang pair.

        Args:
            df_historical: Historical data (date + metric column).
            metric: 'sales' or 'cogs'.
            id_region: Region ID.
            id_gudang: Warehouse ID.

        Returns:
            DataFrame with date, id_region, id_gudang, and metric forecast columns.
        """
        key = f"{metric}_r{id_region}_g{id_gudang}"
        model = self.models.get(key)
        if model is None:
            logger.error("No trained model found for %s", key)
            return None

        model.eval()
        scaler = self.scalers[key]
        s_scaler = self.seasonal_scalers[f"{key}_seasonal"]

        # Build forecast date list (business days only)
        start = df_historical["date"].max() + timedelta(days=1)
        forecast_dates: list[pd.Timestamp] = []
        current = start
        while len(forecast_dates) < self.config.forecast_horizon:
            if self.is_valid_business_day(current):
                forecast_dates.append(current)
            current += timedelta(days=1)

        last_seq_raw = df_historical.iloc[-self.config.input_window :][metric].values.reshape(-1, 1)
        last_seq_scaled = scaler.transform(last_seq_raw)
        current_seq = torch.FloatTensor(last_seq_scaled).unsqueeze(0).to(self.device)

        preds_scaled: list[float] = []
        with torch.no_grad():
            for date in forecast_dates:
                sf = self.create_seasonal_features([date])
                sf_scaled = s_scaler.transform(sf)
                sf_tensor = torch.FloatTensor(sf_scaled).to(self.device)
                pred = model(current_seq, sf_tensor)
                preds_scaled.append(pred.item())
                new_seq_np = current_seq.squeeze(0).cpu().numpy().copy()
                new_seq_np = np.roll(new_seq_np, -1, axis=0)
                new_seq_np[-1, 0] = pred.item()
                current_seq = torch.FloatTensor(new_seq_np).unsqueeze(0).to(self.device)

        preds_orig = scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        preds_orig = np.clip(preds_orig, 0, None)

        return pd.DataFrame(
            {
                "date": forecast_dates,
                "id_region": id_region,
                "id_gudang": id_gudang,
                metric: preds_orig,
            }
        )

    def run_forecasting(
        self,
        data_dict: dict[str, pd.DataFrame],
    ) -> dict[str, dict]:
        """Train and forecast for all metrics and region-gudang pairs.

        Args:
            data_dict: Dict with 'sales' and 'cogs' DataFrames.

        Returns:
            Dict keyed by model_key with 'forecast_df', 'historical_df',
            'metric', 'id_region', 'id_gudang'.
        """
        results: dict[str, dict] = {}

        for metric, df in data_dict.items():
            pairs = df[["id_region", "id_gudang"]].drop_duplicates().values
            logger.info("Training %d models for metric=%s", len(pairs), metric)

            for id_region, id_gudang in pairs:
                key = f"{metric}_r{id_region}_g{id_gudang}"
                X, y, seasonal, processed = self.prepare_data_with_seasonality(
                    df, metric, id_region, id_gudang
                )
                if X is None:
                    continue

                self.train_model(X, y, seasonal, key)
                forecast_df = self.generate_seasonal_forecast(
                    processed, metric, id_region, id_gudang
                )
                if forecast_df is None:
                    continue

                results[key] = {
                    "forecast_df": forecast_df,
                    "historical_df": processed,
                    "metric": metric,
                    "id_region": id_region,
                    "id_gudang": id_gudang,
                }

        logger.info("Individual forecasting complete. %d models trained.", len(results))
        return results
