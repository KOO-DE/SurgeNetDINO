# ---------------------------------------------------------------
# TEP training curve callback
# CSV로 저장된 metrics를 읽어 epoch마다 PNG 커브 플롯 저장
# ---------------------------------------------------------------

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import lightning


class TrainingCurvePlotCallback(lightning.Callback):
    """
    CSVLogger가 저장하는 metrics.csv를 읽어서
    매 validation epoch마다 학습 커브 PNG를 저장하는 콜백.

    저장되는 플롯:
        {save_dir}/curves/loss_curve.png        - train loss
        {save_dir}/curves/miou_curve.png        - val mIoU
        {save_dir}/curves/pixel_acc_curve.png   - val pixel accuracy
    """

    def __init__(self, save_dir: str = "curves"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _read_csv(self, csv_path: Path):
        """metrics.csv → {column: [값들]} 딕셔너리로 변환"""
        if not csv_path.exists():
            return {}

        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return {}

        data = {}
        for key in rows[0].keys():
            vals = []
            for row in rows:
                v = row[key].strip()
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(None)
            data[key] = vals

        return data

    def _get_csv_path(self, trainer):
        """CSVLogger의 metrics.csv 경로 반환"""
        if trainer.logger is None:
            return None
        log_dir = getattr(trainer.logger, "log_dir", None)
        if log_dir is None:
            return None
        return Path(log_dir) / "metrics.csv"

    def _filter_valid(self, steps, values):
        """None 제거 후 (step, value) 쌍 반환"""
        pairs = [(s, v) for s, v in zip(steps, values) if s is not None and v is not None]
        if not pairs:
            return [], []
        s, v = zip(*pairs)
        return list(s), list(v)

    def _plot_and_save(self, trainer):
        csv_path = self._get_csv_path(trainer)
        if csv_path is None:
            return

        data = self._read_csv(csv_path)
        if not data:
            return

        epochs = data.get("epoch", [])

        # --- 1. Loss curve ---
        loss_key = "losses/train_loss_total"
        if loss_key in data:
            steps, vals = self._filter_valid(epochs, data[loss_key])
            if steps:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(steps, vals, marker="o", markersize=3, label="train loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Train Loss")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.save_dir / "loss_curve.png", dpi=150)
                plt.close(fig)

        # --- 2. mIoU curve (val) ---
        # EoMT semantic: metrics/val_iou_all (마지막 block)
        miou_keys = [k for k in data.keys() if "val_iou_all" in k]
        if miou_keys:
            # 마지막 block (suffix 없는 것 또는 가장 마지막) 우선
            main_key = next((k for k in miou_keys if not k.endswith(("_0", "_1", "_2", "_3"))), miou_keys[-1])
            steps, vals = self._filter_valid(epochs, data[main_key])
            if steps:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(steps, [v * 100 for v in vals], marker="o", markersize=3,
                        color="steelblue", label="val mIoU")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("mIoU (%)")
                ax.set_title("Validation mIoU")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.save_dir / "miou_curve.png", dpi=150)
                plt.close(fig)

        # --- 3. Pixel Accuracy curve (val) ---
        acc_keys = [k for k in data.keys() if "val_pixel_acc" in k or "val_acc" in k]
        if acc_keys:
            main_key = acc_keys[-1]
            steps, vals = self._filter_valid(epochs, data[main_key])
            if steps:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(steps, [v * 100 for v in vals], marker="o", markersize=3,
                        color="darkorange", label="val pixel acc")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Pixel Accuracy (%)")
                ax.set_title("Validation Pixel Accuracy")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.save_dir / "pixel_acc_curve.png", dpi=150)
                plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._plot_and_save(trainer)