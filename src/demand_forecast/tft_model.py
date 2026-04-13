import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    Baseline,
)

from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import RMSE

import mlflow

def prepare_tft_data(feature_path:str = "data/processed/feature_table.parquet"):
    df = pd.read_parquet(feature_path)
    mask = (df["state_id"] == "CA") & (df["store_id"].isin(["CA_1"]))
    filtered = df[mask]

    top_items = (
        filtered.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .head(100)
        .index.to_list()
    )

    df = filtered[filtered['item_id'].isin(top_items)].copy()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id', 'item_id', 'date'])

    min_date = df['date'].min()
    df['time_idx'] = (df['date'] - min_date).dt.days

    df["series_id"] = df["store_id"] + "_" + df["item_id"]
    for col in ["item_id", "store_id", "dept_id", "cat_id", "state_id", "series_id"]:
        df[col] = df[col].astype(str)

    df["has_event"] = df["has_event"].fillna(0).astype(int).astype(str)
    df["snap_flag"] = df["snap_flag"].fillna(0).astype(int).astype(str) \
        if "snap_flag" in df.columns else "0"
    
    df["sales"] = df["sales"].astype(float)
    df["sell_price_missing"] = df["sell_price"].isna().astype(int).astype(str) 
    
    df["sell_price"] = (
        df.groupby("series_id")["sell_price"] 
        .ffill() 
        .bfill() 
    )

    df['day_of_week'] = df['day_of_week'].astype(int).astype(str)
    df['month'] = df['month'].astype(int).astype(str)

    return df

def create_tft_datasets(df: pd.DataFrame):
    #4 months of history
    max_encoder_length = 120
    max_prediction_length = 28

    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx = 'time_idx',
        target = 'sales',
        group_ids = ['series_id'],

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        static_categoricals=["store_id", "item_id", "dept_id", "cat_id", "state_id"],
        time_varying_known_categoricals=["has_event", "day_of_week", "month", "sell_price_missing"],
        time_varying_known_reals=["time_idx", "sell_price"],

        time_varying_unknown_reals=["sales", ],

        target_normalizer=GroupNormalizer(
            groups=["series_id"],
            transformation="log1p",
        ),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict = True,
        stop_randomization = True
    )

    print(f"Training samples: {len(training):,}")
    print(f"Validation samples: {len(validation):,}")

    return training, validation

def train_tft(
        training:TimeSeriesDataSet,
        val:TimeSeriesDataSet,
        max_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.001
):
    
    train_loader = training.to_dataloader(
        train = True,
        batch_size = batch_size,
        num_workers = 2,
        persistent_workers=True,
        pin_memory = True
    )

    val_dataloader = val.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,
        pin_memory = True
    )

    early_stop = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        min_delta = 0.0001,
        verbose = True,
        mode = 'min'
    )

    lr_monitor = LearningRateMonitor()

    logger = TensorBoardLogger("tb_logs", name="tft")

    trainer = pl.Trainer(
        max_epochs = max_epochs,
        accelerator = 'gpu',
        devices = 1,
        callbacks=[early_stop, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate = learning_rate,
        hidden_size = 32,
        attention_head_size = 4,
        dropout = 0.2,
        hidden_continuous_size = 32,
        loss = RMSE(),
        reduce_on_plateau_patience=4,
    )

    print(f"TFT parameters: {tft.size()/1e3:.1f}k")

    with mlflow.start_run(run_name="tft-v1"):
        mlflow.log_params({
            "model": "TFT",
            "hidden_size": 32,
            "attention_heads": 4,
            "accelerator": "gpu",
            "gpu": "RTX 4070",
            "max_encoder_length": training.max_encoder_length,
            "max_prediction_length": training.max_prediction_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_series": len(training.decoded_index["series_id"].unique()),
        })

        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_dataloader)

        # Log best validation loss
        mlflow.log_metric("best_val_loss", trainer.callback_metrics["val_loss"].item())
    
    return tft, trainer
    
    
def evaluate_tft(tft, validation, val_dataloader):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs("reports", exist_ok=True)

    # --- Predictions ---
    predictions = tft.predict(
        val_dataloader,
        return_x=True,
        return_index=True,
    )

    # --- Raw predictions for plotting / interpretation ---
    raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True)

    # --- Example prediction plots ---
    for idx in range(min(4, len(predictions.output))):
        fig = tft.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=idx,
            add_loss_to_title=True,
        )
        fig.savefig(f"reports/tft_prediction_{idx}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # --- Feature importance / interpretation ---
    interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")

    figs = tft.plot_interpretation(interpretation)

    if isinstance(figs, dict):
        for name, fig in figs.items():
            fig.savefig(f"reports/tft_interpretation_{name}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
    else:
        figs.savefig("reports/tft_interpretation.png", dpi=100, bbox_inches="tight")
        plt.close(figs)

    print("\nInterpretation saved to reports/")
    print(f"Static variable importance: {interpretation['static_variables']}")
    print(f"Encoder variable importance: {interpretation['encoder_variables']}")
    print(f"Decoder variable importance: {interpretation['decoder_variables']}")

    return predictions, interpretation

    


if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)

    print("Step 1: Preparing data...")
    df = prepare_tft_data()

    print("\nStep 2: Creating datasets...")
    training, validation = create_tft_datasets(df)

    print("\nStep 3: Training TFT...")
    tft, trainer = train_tft(training, validation)

    print("\nStep 4: Evaluating...")
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=64, num_workers=0
    )
    predictions, interpretation = evaluate_tft(tft, validation, val_dataloader)

    print("\nDone! Check reports/ for plots and tb_logs/ for TensorBoard logs.")
    print("Run 'tensorboard --logdir tb_logs' to view training curves.")


