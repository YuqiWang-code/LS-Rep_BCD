"""Plain-text logging utilities for Run3 BT-SAM-RDT.

The logger is intentionally simple and auditable.

Run3 logging contract
---------------------
1. Training configuration is written at logger construction time.
2. One compact validation/training summary is written per epoch.
3. Arbitrary diagnostic messages can be appended through ``log_message``.
4. The authoritative final test block is written by ``models/scripts/train.py``
   through repeated ``log_message`` calls between:

       === TEST RESULTS ===
       === END TEST RESULTS ===

This module does not interpret experiment results and does not select or delete
checkpoints. Checkpoint retention policy remains in ``models/scripts/train.py``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping


LOG_WIDTH = 100
LOG_TITLE = "A2Net-LWGANet-L0 Run3 BT-SAM-RDT Training Log"


# ============================================================================
# Formatting helpers
# ============================================================================


def _format_config_value(
    value: Any,
) -> str:
    """Convert one config value to a stable single-line representation."""
    text = str(
        value
    )

    # Keep the configuration header line-oriented even if a future value
    # contains a newline.
    return text.replace(
        "\n",
        "\\n",
    )


def _format_metric(
    value: Any,
    decimals: int = 4,
) -> str:
    """Format numeric-like values used in epoch summaries."""
    try:
        return f"{float(value):.{decimals}f}"

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            f"Metric value must be numeric, got {value!r}"
        ) from exc


# ============================================================================
# Training logger
# ============================================================================


class TrainingLogger:
    """Small append-safe text logger used by the Run3 trainer.

    Parameters
    ----------
    log_file:
        Destination ``train_log.txt`` path.

    config:
        Mapping containing the complete experiment/runtime configuration.

    append:
        ``False`` for a new run.
        ``True`` when resuming into an existing log.
    """

    def __init__(
        self,
        log_file,
        config,
        append: bool = False,
    ) -> None:
        self.log_file = str(
            log_file
        )

        self.config = (
            dict(
                config
            )
            if isinstance(
                config,
                Mapping,
            )
            else None
        )

        if self.config is None:
            raise TypeError(
                "config must be a mapping"
            )

        self.start_time = time.time()

        path = Path(
            self.log_file
        )

        if (
            not path.name
            or path.name in {
                ".",
                "..",
            }
        ):
            raise ValueError(
                f"Invalid log path: {path}"
            )

        # Path.parent is "." for a bare file name, so this is safe unlike
        # os.makedirs(os.path.dirname(...)) when dirname == "".
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_header(
            append=bool(
                append
            )
        )

    # ======================================================================
    # Internal I/O
    # ======================================================================

    def _append_line(
        self,
        message: Any,
    ) -> None:
        """Append one logical line and flush it to disk."""
        text = str(
            message
        )

        # Preserve caller-provided multi-line messages as text, but guarantee
        # exactly one trailing newline for the append operation.
        text = text.rstrip(
            "\n"
        )

        with open(
            self.log_file,
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                text
                + "\n"
            )

            handle.flush()

            try:
                os.fsync(
                    handle.fileno()
                )

            except OSError:
                # Logging should not fail solely because a filesystem does not
                # support fsync in the current environment.
                pass

    def _write_header(
        self,
        append: bool,
    ) -> None:
        """Write a new-run or resume header."""
        path = Path(
            self.log_file
        )

        mode = (
            "a"
            if append
            else "w"
        )

        with open(
            path,
            mode,
            encoding="utf-8",
        ) as handle:
            if append:
                # Make the resume boundary visually explicit without
                # pretending it is a fresh independent run.
                handle.write(
                    "\n"
                    + "=" * LOG_WIDTH
                    + "\n"
                )

                handle.write(
                    "RUN RESUME\n"
                )

                handle.write(
                    "=" * LOG_WIDTH
                    + "\n"
                )

            else:
                handle.write(
                    "=" * LOG_WIDTH
                    + "\n"
                )

                handle.write(
                    LOG_TITLE
                    + "\n"
                )

                handle.write(
                    "=" * LOG_WIDTH
                    + "\n"
                )

            for (
                key,
                value,
            ) in self.config.items():
                handle.write(
                    f"{key}: "
                    f"{_format_config_value(value)}"
                    "\n"
                )

            handle.write(
                "-" * LOG_WIDTH
                + "\n"
            )

            handle.flush()

            try:
                os.fsync(
                    handle.fileno()
                )

            except OSError:
                pass

    # ======================================================================
    # Public API
    # ======================================================================

    def log_epoch(
        self,
        epoch,
        total_epochs,
        train_losses,
        val_metrics,
        lr,
        gpu_mem,
        is_best,
    ) -> None:
        """Write one Run3 epoch summary.

        This signature intentionally matches the existing trainer.

        ``train_losses`` may contain the full Run3 diagnostic dictionary
        (main loss, KD loss, Fast Teacher statistics, BT-SAM matching,
        foundation-prior diagnostics, etc.). Fields are emitted in the mapping
        order supplied by ``train.py``.

        ``val_metrics`` is expected to provide:
            f1
            iou
            recall
            precision
            kappa
            oa
        """
        if not isinstance(
            train_losses,
            Mapping,
        ):
            raise TypeError(
                "train_losses must be a mapping"
            )

        if not isinstance(
            val_metrics,
            Mapping,
        ):
            raise TypeError(
                "val_metrics must be a mapping"
            )

        required_validation_metrics = (
            "f1",
            "iou",
            "recall",
            "precision",
            "kappa",
            "oa",
        )

        missing = [
            key
            for key in required_validation_metrics
            if key not in val_metrics
        ]

        if missing:
            raise KeyError(
                "val_metrics missing fields: "
                + ", ".join(
                    missing
                )
            )

        loss_parts = []

        for (
            key,
            value,
        ) in train_losses.items():
            loss_parts.append(
                f"{key}="
                f"{_format_metric(value, 6)}"
            )

        losses = " ".join(
            loss_parts
        )

        line = (
            f"Epoch [{int(epoch)}/{int(total_epochs)}] "
            f"{losses} "
            f"F1={_format_metric(val_metrics['f1'], 6)} "
            f"IoU={_format_metric(val_metrics['iou'], 6)} "
            f"Recall={_format_metric(val_metrics['recall'], 6)} "
            f"Precision={_format_metric(val_metrics['precision'], 6)} "
            f"OA={_format_metric(val_metrics['oa'], 6)} "
            f"Kappa={_format_metric(val_metrics['kappa'], 6)} "
            f"LR={_format_metric(lr, 8)} "
            f"GPU={_format_metric(gpu_mem, 3)}GB"
        )

        if bool(
            is_best
        ):
            line += " BEST"

        print(
            line,
            flush=True,
        )

        self.log_message(
            line
        )

    def log_message(
        self,
        message,
    ) -> None:
        """Append an arbitrary audit/result message."""
        self._append_line(
            message
        )


__all__ = [
    "TrainingLogger",
]
