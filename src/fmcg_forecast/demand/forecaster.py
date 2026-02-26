"""Demand forecasting: training and inference with BiLSTM-Attention.

Wraps the shared TimeSeriesLSTMAttentionModel with:
- STL-based de-trending
- Time-series cross-validation
- MinMaxScaler normalization
- Trend extrapolation for future predictions
"""

import copy
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from statsmodels.tsa.seasonal import STL

from fmcg_forecast.config import DemandConfig
from fmcg_forecast.data.calendar import FMCG_HOLIDAYS
from fmcg_forecast.models.lstm_attention import (
    QuantileLoss,
    TimeSeriesLSTMAttentionModel,
)


logger = logging.getLogger(__name__)

_LEBARAN_DATES: list[datetime] = [
    datetime(y, d.month, d.day) for y, d in FMCG_HOLIDAYS["idul_fitri"].items()
]


class TimeSeriesForecaster:
    """Global LSTM-Attention demand forecaster for FMCG products.

    Trains a single shared model across all product-warehouse combinations
    using categorical embeddings for product and warehouse identity.
    """

    def __init__(self, config: DemandConfig) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.scalers: dict[str, MinMaxScaler] = {}
        self.trends: dict[str, pd.DataFrame] = {}
        self.spike_factors: dict[str, float] = {}
        self.global_model: TimeSeriesLSTMAttentionModel | None = None
        self.feature_names: list[str] = []
        self.product_encoder = LabelEncoder()
        self.gudang_encoder = LabelEncoder()
        self.holidays: set[date] = set()
        logger.info("TimeSeriesForecaster initialized on device: %s", self.device)

    def save_state(self, path: str | Path) -> None:
        """Save model weights, scalers, and encoders to disk."""
        if self.global_model is None:
            logger.error("Model not trained — cannot save state.")
            return
        state = {
            "model_state_dict": self.global_model.state_dict(),
            "config": self.config,
            "scalers": self.scalers,
            "trends": self.trends,
            "product_encoder": self.product_encoder,
            "gudang_encoder": self.gudang_encoder,
            "feature_names": self.feature_names,
            "spike_factors": self.spike_factors,
            "holidays": self.holidays,
        }
        torch.save(state, path)
        logger.info("Model state saved to %s", path)

    def load_state(self, path: str | Path) -> bool:
        """Load a previously saved model state.

        Returns:
            True if loaded successfully, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(
                "No model state found at %s — will train from scratch.", path
            )
            return False
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.config = state["config"]
            self.scalers = state["scalers"]
            self.trends = state["trends"]
            self.product_encoder = state["product_encoder"]
            self.gudang_encoder = state["gudang_encoder"]
            self.feature_names = state["feature_names"]
            self.spike_factors = state.get("spike_factors", {})
            self.holidays = state.get("holidays", set())
            self.global_model = self._instantiate_model()
            self.global_model.load_state_dict(state["model_state_dict"])
            logger.info("Model state loaded from %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to load model state: %s — training from scratch.", exc)
            return False

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag, rolling, and Lebaran proximity features."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        sales = df["main_product_sales"]
        for lag in [1, 7, 14]:
            df[f"sales_lag_{lag}"] = sales.shift(lag)
        for window in [7, 14]:
            df[f"sales_rolling_mean_{window}"] = sales.rolling(window=window).mean()
            df[f"sales_rolling_std_{window}"] = sales.rolling(window=window).std()

        # Lebaran proximity signal
        df["_days_until_leb"] = 100.0
        for hol_dt in _LEBARAN_DATES:
            delta = (pd.Timestamp(hol_dt) - df["date"]).dt.days
            df["_days_until_leb"] = np.minimum(
                df["_days_until_leb"], delta.where((delta >= 0) & (delta <= 60), 100)
            )
        df["lebaran_proximity_signal"] = 1 - (df["_days_until_leb"] / 60)
        df.loc[df["_days_until_leb"] > 60, "lebaran_proximity_signal"] = 0
        df.drop(columns=["_days_until_leb"], inplace=True)

        if "promo_sales" in df.columns and "is_payday_period" in df.columns:
            df["promo_on_payday"] = df["promo_sales"] * df["is_payday_period"]

        df.bfill(inplace=True)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        return df

    def prepare_data_for_series(
        self, df: pd.DataFrame, unique_key: str
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """Prepare windowed tensor datasets for a single product-warehouse series.

        Args:
            df: Series DataFrame with date + feature columns.
            unique_key: Identifier string (product_warehouse).

        Returns:
            (X_tensor, y_tensor, processed_df) tuple.
        """
        processed = self._create_features(df)

        # STL de-trending
        stl_period = 365 if len(processed) >= 730 else max(7, len(processed) // 4)
        if len(processed["main_product_sales"]) < 2 * stl_period:
            processed["trend"] = (
                processed["main_product_sales"]
                .rolling(window=7, center=True)
                .mean()
                .bfill()
                .ffill()
            )
        else:
            res = STL(
                processed["main_product_sales"], period=stl_period, robust=True
            ).fit()
            processed["trend"] = res.trend

        processed["main_product_sales_detrended"] = (
            processed["main_product_sales"] - processed["trend"]
        )
        self.trends[unique_key] = processed[["date", "trend"]].set_index("date")

        feature_cols = [
            "main_product_sales_detrended",
            "trend",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_rolling_mean_7",
            "sales_rolling_std_7",
            "prioritas_sales",
            "promo_sales",
            "lebaran_proximity_signal",
            "promo_on_payday",
            "is_payday_period",
        ]
        if not self.feature_names:
            self.feature_names = feature_cols
        for col in self.feature_names:
            if col not in processed.columns:
                processed[col] = 0

        scaler = MinMaxScaler()
        processed[self.feature_names] = scaler.fit_transform(
            processed[self.feature_names]
        )
        self.scalers[unique_key] = scaler

        horizon = self.config.forecast_horizon
        window = self.config.input_window
        X, y = [], []
        for i in range(len(processed) - window - horizon + 1):
            X.append(processed.iloc[i : i + window][self.feature_names].values)
            y.append(
                processed.iloc[i + window : i + window + horizon][
                    "main_product_sales_detrended"
                ].values
            )

        if not X:
            empty = torch.zeros(0)
            return empty, empty, processed

        X_t = torch.FloatTensor(np.array(X)).to(self.device)
        y_t = torch.FloatTensor(np.array(y)).to(self.device)
        return X_t, y_t, processed

    def _train_single_fold(
        self,
        model: TimeSeriesLSTMAttentionModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        x_cat_prod_train: torch.Tensor,
        x_cat_prod_val: torch.Tensor,
        x_cat_gudang_train: torch.Tensor,
        x_cat_gudang_val: torch.Tensor,
    ) -> None:
        """Train model for one fold with early stopping."""
        criterion = QuantileLoss(self.config.quantiles)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.model_params.learning_rate,
            weight_decay=self.config.model_params.weight_decay,
        )
        patience = self.config.model_params.early_stopping_patience
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state: dict | None = None

        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0.0
            bs = self.config.batch_size
            for i in range(0, len(X_train), bs):
                bX = X_train[i : i + bs]
                by = y_train[i : i + bs]
                bp = x_cat_prod_train[i : i + bs].unsqueeze(1)
                bg = x_cat_gudang_train[i : i + bs].unsqueeze(1)
                optimizer.zero_grad()
                out = model(bX, bp, bg)
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_out = model(
                    X_val, x_cat_prod_val.unsqueeze(1), x_cat_gudang_val.unsqueeze(1)
                )
                val_loss = criterion(val_out, y_val).item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.debug(
                    "Epoch %d/%d — val_loss=%.6f",
                    epoch + 1,
                    self.config.epochs,
                    val_loss,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.6f)",
                    epoch + 1,
                    best_val_loss,
                )
                break

        if best_state:
            model.load_state_dict(best_state)

    def train_global_model(self, df: pd.DataFrame) -> None:
        """Train the global LSTM-Attention model on all product-warehouse series.

        Args:
            df: Full demand DataFrame with date, product_name, id_gudang columns.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        self.product_encoder.fit(df["product_name"].unique())
        self.gudang_encoder.fit(df["id_gudang"].unique())
        logger.info(
            "Fitting model on %d products x %d warehouses",
            len(self.product_encoder.classes_),
            len(self.gudang_encoder.classes_),
        )

        # Filter business days (Mon-Sat)
        df_biz = df[df["date"].dt.dayofweek < 6].copy()

        all_X: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []
        all_prod: list[int] = []
        all_gudang: list[int] = []

        for (product, gudang), grp in df_biz.groupby(["product_name", "id_gudang"]):
            key = f"{product}_{gudang}"
            X, y, _ = self.prepare_data_for_series(grp.reset_index(drop=True), key)
            if X.shape[0] > 0:
                all_X.append(X)
                all_y.append(y)
                pid = int(self.product_encoder.transform([product])[0])
                gid = int(self.gudang_encoder.transform([gudang])[0])
                all_prod.extend([pid] * X.shape[0])
                all_gudang.extend([gid] * X.shape[0])

        if not all_X:
            logger.error("No training sequences could be prepared. Aborting.")
            return

        X_global = torch.cat(all_X, dim=0)
        y_global = torch.cat(all_y, dim=0)
        cat_prod = torch.LongTensor(all_prod).to(self.device)
        cat_gudang = torch.LongTensor(all_gudang).to(self.device)
        logger.info("Total training sequences: %d", len(X_global))

        if self.global_model is None:
            self.global_model = self._instantiate_model()

        # Time series cross-validation — use the last fold only
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        splits = list(tscv.split(X_global))
        train_idx, val_idx = splits[-1]

        self._train_single_fold(
            self.global_model,
            X_global[train_idx],
            y_global[train_idx],
            X_global[val_idx],
            y_global[val_idx],
            cat_prod[train_idx],
            cat_prod[val_idx],
            cat_gudang[train_idx],
            cat_gudang[val_idx],
        )
        logger.info("Global model training complete.")

    def _instantiate_model(self) -> TimeSeriesLSTMAttentionModel:
        mp = self.config.model_params
        return TimeSeriesLSTMAttentionModel(
            input_dim=len(self.feature_names),
            hidden_dim=mp.hidden_dim,
            num_layers=mp.num_layers,
            output_dim=self.config.forecast_horizon,
            dropout_prob=mp.dropout,
            num_quantiles=len(self.config.quantiles),
            num_product_embeddings=len(self.product_encoder.classes_),
            product_embedding_dim=mp.product_embedding_dim,
            num_gudang_embeddings=len(self.gudang_encoder.classes_),
            gudang_embedding_dim=mp.gudang_embedding_dim,
        ).to(self.device)

    def predict_future(
        self, historical_df: pd.DataFrame, forecast_start_date: str
    ) -> pd.DataFrame:
        """Generate demand forecasts for all product-warehouse combinations.

        Args:
            historical_df: Historical demand data.
            forecast_start_date: First forecast date (YYYY-MM-DD).

        Returns:
            DataFrame with forecast_date, main_product_name, id_gudang, forecast_value.
        """
        if self.global_model is None:
            logger.error("Model not trained — cannot predict.")
            return pd.DataFrame()

        self.global_model.eval()
        historical_df = historical_df.copy()
        historical_df["date"] = pd.to_datetime(historical_df["date"])
        historical_df["main_product_sales"] = np.log1p(
            historical_df["main_product_sales"]
        )

        all_forecasts = []
        groups = list(historical_df.groupby(["product_name", "id_gudang"]))
        logger.info("Generating forecasts for %d product-warehouse pairs", len(groups))

        for (product, warehouse), grp in groups:
            key = f"{product}_{warehouse}"
            if key not in self.scalers:
                continue

            _, _, processed = self.prepare_data_for_series(
                grp.reset_index(drop=True), key
            )
            last_window = processed.tail(self.config.input_window)[
                self.feature_names
            ].values.copy()
            x_cont = torch.FloatTensor(last_window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pid = int(self.product_encoder.transform([product])[0])
                gid = int(self.gudang_encoder.transform([warehouse])[0])
                x_prod = torch.LongTensor([[pid]]).to(self.device)
                x_gudang = torch.LongTensor([[gid]]).to(self.device)
                predicted = self.global_model(x_cont, x_prod, x_gudang)

            q_idx = self.config.quantiles.index(0.5)
            median_scaled = predicted[0, :, q_idx].cpu().numpy()
            median_detrended = self._inverse_transform(median_scaled, self.scalers[key])
            trend_extrap = self._extrapolate_trend(
                self.trends[key]["trend"].dropna(), self.config.forecast_horizon
            )
            base_forecast = np.expm1(median_detrended + trend_extrap)

            forecast_dates = pd.bdate_range(
                start=forecast_start_date, periods=self.config.forecast_horizon
            )
            forecast_values = np.ceil(np.maximum(0, base_forecast))
            all_forecasts.append(
                pd.DataFrame(
                    {
                        "forecast_date": forecast_dates,
                        "main_product_name": product,
                        "id_gudang": warehouse,
                        "forecast_value": forecast_values,
                    }
                )
            )

        logger.info("Forecast generation complete.")
        return (
            pd.concat(all_forecasts, ignore_index=True)
            if all_forecasts
            else pd.DataFrame()
        )

    def _extrapolate_trend(self, trend_series: pd.Series, horizon: int) -> np.ndarray:
        """Linear trend extrapolation for the forecast horizon."""
        recent = min(365, len(trend_series))
        if recent < 2:
            return np.zeros(horizon)
        x = np.arange(recent)
        y = trend_series.iloc[-recent:].values
        coeffs = np.polyfit(x, y, 1)
        return np.polyval(coeffs, np.arange(recent, recent + horizon))

    def _inverse_transform(
        self, series_scaled: np.ndarray, scaler: MinMaxScaler
    ) -> np.ndarray:
        """Inverse-transform the first feature column."""
        placeholder = np.zeros((len(series_scaled), scaler.n_features_in_))
        placeholder[:, 0] = series_scaled.flatten()
        return scaler.inverse_transform(placeholder)[:, 0]
