"""Merged/aggregated sales and COGS forecaster.

Rewrites yokulak-forecasting/sales/merged/forecasting.py without
any MySQL/database dependencies.

COGS model depends sequentially on the sales forecast (uses predicted
sales as an input feature during autoregressive generation).
"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from fmcg_forecast.config import SalesConfig
from fmcg_forecast.models.lstm_attention import FinancialLSTMModel


logger = logging.getLogger(__name__)

# Day-of-week one-hot columns (Mon=0 … Sun=6)
_DAY_COLS = [f"day_{i}" for i in range(7)]


class FinancialForecaster:
    """Plain LSTM forecaster for aggregated (total) sales and COGS.

    Training order is sequential: sales model trained first, then COGS
    model uses predicted sales as an additional input feature.
    """

    def __init__(self, config: SalesConfig, holiday_dates: set) -> None:
        self.config = config
        self.holidays = holiday_dates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers: dict[str, MinMaxScaler] = {
            "sales": MinMaxScaler(),
            "cogs": MinMaxScaler(),
            "sales_for_cogs": MinMaxScaler(),
        }
        self.models: dict[str, FinancialLSTMModel | None] = {
            "sales": None,
            "cogs": None,
        }
        self.sales_forecast_df: pd.DataFrame | None = None
        logger.info("FinancialForecaster using device: %s", self.device)

    def is_valid_business_day(self, d: pd.Timestamp) -> bool:
        """Return True if d is not a Sunday and not a holiday."""
        return d.weekday() != 6 and d.date() not in self.holidays

    def _add_day_of_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot day-of-week columns (day_0 … day_6) to df."""
        df = df.copy()
        df["_dow"] = df["date"].dt.dayofweek
        dummies = pd.get_dummies(df["_dow"], prefix="day", drop_first=False, dtype=int)
        # Ensure all 7 columns are present even if some DOWs are absent
        for col in _DAY_COLS:
            if col not in dummies.columns:
                dummies[col] = 0
        df = pd.concat([df, dummies[_DAY_COLS]], axis=1)
        df.drop(columns=["_dow"], inplace=True)
        return df

    def _get_one_hot_day(self, d: pd.Timestamp) -> list[int]:
        """Return 7-element one-hot for day-of-week."""
        oh = [0] * 7
        oh[d.weekday()] = 1
        return oh

    def prepare_data_sales(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, pd.DataFrame]:
        """Prepare (X, y) sequences for the sales model.

        Input features per timestep: [scaled_sales, day_0..day_6] = 8 dims.

        Args:
            df: DataFrame with date and sales columns.

        Returns:
            Tuple of (X, y, processed_df).
        """
        df = df.sort_values("date").ffill().fillna(0).copy()
        df = self._add_day_of_week(df)

        scaled_sales = self.scalers["sales"].fit_transform(df[["sales"]].values)
        combined = np.hstack((scaled_sales, df[_DAY_COLS].values)).astype(np.float32)

        X_list, y_list = [], []
        for i in range(len(combined) - self.config.input_window):
            X_list.append(combined[i : i + self.config.input_window])
            y_list.append(scaled_sales[i + self.config.input_window, 0])

        if not X_list:
            return None, None, df

        X = torch.FloatTensor(np.array(X_list)).to(self.device)
        y = torch.FloatTensor(np.array(y_list).reshape(-1, 1)).to(self.device)
        return X, y, df

    def prepare_data_cogs(
        self,
        sales_df: pd.DataFrame,
        cogs_df: pd.DataFrame,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, pd.DataFrame]:
        """Prepare (X, y) sequences for the COGS model.

        Input features: [scaled_sales, scaled_cogs, day_0..day_6] = 9 dims.

        Args:
            sales_df: DataFrame with date and sales columns.
            cogs_df: DataFrame with date and cogs columns.

        Returns:
            Tuple of (X, y, processed_df).
        """
        df = (
            pd.merge(
                sales_df[["date", "sales"]].copy(),
                cogs_df[["date", "cogs"]].copy(),
                on="date",
                how="inner",
            )
            .sort_values("date")
            .ffill()
            .fillna(0)
            .copy()
        )
        df = self._add_day_of_week(df)

        scaled_sales = self.scalers["sales_for_cogs"].fit_transform(df[["sales"]].values)
        scaled_cogs = self.scalers["cogs"].fit_transform(df[["cogs"]].values)
        combined = np.hstack(
            (scaled_sales, scaled_cogs, df[_DAY_COLS].values)
        ).astype(np.float32)

        X_list, y_list = [], []
        for i in range(len(combined) - self.config.input_window):
            X_list.append(combined[i : i + self.config.input_window])
            y_list.append(scaled_cogs[i + self.config.input_window, 0])

        if not X_list:
            return None, None, df

        X = torch.FloatTensor(np.array(X_list)).to(self.device)
        y = torch.FloatTensor(np.array(y_list).reshape(-1, 1)).to(self.device)
        return X, y, df

    def _train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        metric: str,
    ) -> None:
        """Train a FinancialLSTMModel for the given metric.

        Args:
            X: Input sequences (samples, window, features).
            y: Target values (samples, 1).
            metric: 'sales' or 'cogs'.
        """
        split = max(1, int(len(X) * 0.8))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        if len(X_val) == 0:
            X_val, y_val = X_tr, y_tr

        model = FinancialLSTMModel(
            input_dim=X.shape[2],
            hidden_dim=self.config.model_params.hidden_dim,
            num_layers=self.config.model_params.num_layers,
            output_dim=1,
            dropout_rate=self.config.model_params.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.model_params.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_tr), y_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            scheduler.step(val_loss)

        self.models[metric] = model
        logger.info("Trained %s model. Final val_loss=%.6f", metric, val_loss)

    def generate_sales_forecast(
        self,
        df: pd.DataFrame,
        forecast_start: pd.Timestamp,
    ) -> pd.DataFrame:
        """Autoregressively forecast sales.

        Args:
            df: Processed DataFrame with date, sales, and day columns.
            forecast_start: First date to forecast from.

        Returns:
            DataFrame with date and sales forecast.
        """
        model = self.models["sales"]
        assert model is not None, "Sales model not trained"
        model.eval()

        last_seq = df.iloc[-self.config.input_window :]
        last_sales_scaled = self.scalers["sales"].transform(last_seq[["sales"]].values)
        last_dow = last_seq[_DAY_COLS].values
        current_seq = torch.FloatTensor(
            np.hstack((last_sales_scaled, last_dow)).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        # Collect forecast business days
        forecast_dates: list[pd.Timestamp] = []
        cur = forecast_start
        while len(forecast_dates) < self.config.forecast_horizon:
            if self.is_valid_business_day(cur):
                forecast_dates.append(cur)
            cur += timedelta(days=1)

        preds_scaled: list[float] = []
        with torch.no_grad():
            for date in forecast_dates:
                pred = model(current_seq)
                preds_scaled.append(pred.item())
                one_hot = self._get_one_hot_day(date)
                new_row = np.array([pred.item()] + one_hot, dtype=np.float32).reshape(1, -1)
                seq_np = current_seq.squeeze(0).cpu().numpy()
                seq_np = np.vstack([seq_np[1:], new_row])
                current_seq = torch.FloatTensor(seq_np).unsqueeze(0).to(self.device)

        preds_orig = self.scalers["sales"].inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        preds_orig = np.clip(preds_orig, 0, None)

        self.sales_forecast_df = pd.DataFrame({"date": forecast_dates, "sales": preds_orig})
        return self.sales_forecast_df

    def generate_cogs_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Autoregressively forecast COGS using predicted sales as input.

        Args:
            df: Processed DataFrame with date, sales, cogs, and day columns.

        Returns:
            DataFrame with date and cogs forecast.
        """
        assert self.sales_forecast_df is not None, "Sales forecast must be generated first"
        model = self.models["cogs"]
        assert model is not None, "COGS model not trained"
        model.eval()

        last_seq = df.iloc[-self.config.input_window :]
        last_sales_scaled = self.scalers["sales_for_cogs"].transform(last_seq[["sales"]].values)
        last_cogs_scaled = self.scalers["cogs"].transform(last_seq[["cogs"]].values)
        last_dow = last_seq[_DAY_COLS].values
        current_seq = torch.FloatTensor(
            np.hstack((last_sales_scaled, last_cogs_scaled, last_dow)).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        forecast_dates = self.sales_forecast_df["date"].tolist()
        forecast_sales = self.sales_forecast_df["sales"].values

        preds_scaled: list[float] = []
        with torch.no_grad():
            for i, date in enumerate(forecast_dates):
                pred = model(current_seq)
                preds_scaled.append(pred.item())
                next_sales_s = self.scalers["sales_for_cogs"].transform([[forecast_sales[i]]])[0, 0]
                one_hot = self._get_one_hot_day(date)
                new_row = np.array(
                    [next_sales_s, pred.item()] + one_hot, dtype=np.float32
                ).reshape(1, -1)
                seq_np = current_seq.squeeze(0).cpu().numpy()
                seq_np = np.vstack([seq_np[1:], new_row])
                current_seq = torch.FloatTensor(seq_np).unsqueeze(0).to(self.device)

        preds_orig = self.scalers["cogs"].inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        preds_orig = np.clip(preds_orig, 0, None)

        return pd.DataFrame({"date": forecast_dates, "cogs": preds_orig})

    def generate_summary(
        self,
        results: dict[str, dict],
    ) -> pd.DataFrame:
        """Build a summary DataFrame from forecast results.

        Args:
            results: Dict from run_forecasting().

        Returns:
            Summary DataFrame with statistics per metric.
        """
        rows = []
        for metric, data in results.items():
            hist = data["historical_df"]
            fcst = data["forecast_df"]
            hist_mean = hist[metric].mean()
            pct = ((fcst[metric].mean() - hist_mean) / hist_mean * 100) if hist_mean != 0 else 0.0
            rows.append(
                {
                    "metric": metric,
                    "historical_mean": round(hist_mean, 4),
                    "historical_std": round(hist[metric].std(), 4),
                    "forecast_mean": round(fcst[metric].mean(), 4),
                    "forecast_std": round(fcst[metric].std(), 4),
                    "percent_change": round(pct, 2),
                    "data_points": len(hist),
                    "forecast_horizon": len(fcst),
                    "forecast_start_date": fcst["date"].min().date(),
                    "forecast_end_date": fcst["date"].max().date(),
                }
            )
        return pd.DataFrame(rows)

    def run_forecasting(
        self,
        sales_df: pd.DataFrame,
        cogs_df: pd.DataFrame,
    ) -> dict[str, dict]:
        """Run the sequential sales → COGS training and forecasting pipeline.

        Args:
            sales_df: Aggregated daily sales DataFrame (date, sales).
            cogs_df: Aggregated daily COGS DataFrame (date, cogs).

        Returns:
            Dict with 'sales' and 'cogs' keys, each containing
            'forecast_df' and 'historical_df'.
        """
        results: dict[str, dict] = {}

        latest = max(sales_df["date"].max(), cogs_df["date"].max())
        forecast_start = pd.Timestamp(latest) + timedelta(days=1)

        # --- Sales ---
        logger.info("Preparing and training sales model...")
        X_sales, y_sales, processed_sales = self.prepare_data_sales(sales_df)
        if X_sales is not None:
            self._train(X_sales, y_sales, "sales")
            sales_fc = self.generate_sales_forecast(processed_sales, forecast_start)
            results["sales"] = {
                "forecast_df": sales_fc,
                "historical_df": processed_sales,
            }

        # --- COGS (depends on sales) ---
        logger.info("Preparing and training COGS model...")
        X_cogs, y_cogs, processed_merged = self.prepare_data_cogs(sales_df, cogs_df)
        if X_cogs is not None and self.sales_forecast_df is not None:
            self._train(X_cogs, y_cogs, "cogs")
            cogs_fc = self.generate_cogs_forecast(processed_merged)
            results["cogs"] = {
                "forecast_df": cogs_fc,
                "historical_df": processed_merged,
            }

        logger.info("Merged forecasting complete.")
        return results
