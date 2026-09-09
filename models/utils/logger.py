"""Plain-text logger for auditable SAM-HSD training and test results."""

import os
import time


class TrainingLogger:
    def __init__(self, log_file, config, append=False):
        self.log_file = log_file
        self.config = config
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        mode = "a" if append else "w"
        with open(log_file, mode, encoding="utf-8") as handle:
            handle.write("=" * 100 + "\nSAM-HSD Training Log\n" + "=" * 100 + "\n")
            for key, value in config.items():
                handle.write(f"{key}: {value}\n")
            handle.write("-" * 100 + "\n")

    def log_epoch(self, epoch, total_epochs, train_losses, val_metrics, lr,
                  gpu_mem, is_best):
        losses = " ".join(f"{key}={value:.4f}" for key, value in train_losses.items())
        line = (
            f"Epoch [{epoch}/{total_epochs}] {losses} "
            f"F1={val_metrics['f1']:.4f} IoU={val_metrics['iou']:.4f} "
            f"R={val_metrics['recall']:.4f} P={val_metrics['precision']:.4f} "
            f"OA={val_metrics['oa']:.4f} Kappa={val_metrics['kappa']:.4f} "
            f"LR={lr:.7f} GPU={gpu_mem:.2f}GB"
        )
        if is_best:
            line += " BEST"
        print(line)
        self.log_message(line)

    def log_message(self, message):
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")
