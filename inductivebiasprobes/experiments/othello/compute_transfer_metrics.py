import json
import logging

import numpy as np

from inductivebiasprobes.paths import OTHELLO_CKPT_DIR, OTHELLO_EXT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_TYPES = [
    "rnn",
    "lstm",
    "gpt",
    "mamba",
    "mamba2",
]
TRANSFER_TYPES = ["next_token", "scratch"]
TASKS = ["majority", "balance-black", "edges"]


def load_data():
    data_type = "synthetic_othello"
    data = {
        task: {
            model: {
                transfer: {
                    "ib_ss": None,
                    "loss_ds": None,
                    "val_loss": None,
                    "val_accuracy": None,
                }
                for transfer in TRANSFER_TYPES
            }
            for model in MODEL_TYPES
        }
        for task in TASKS
    }

    for model in MODEL_TYPES:
        for transfer in TRANSFER_TYPES:
            # Load IB ratios
            ib_path = (
                OTHELLO_EXT_DIR
                / data_type
                / "white_noise"
                / model
                / f"pt_{transfer}"
                / "100_examples"
                / "100_iters"
            )
            with open(ib_path / "ib.json") as f:
                result = json.load(f)
                ib_ss = result["same_state_ib"]
                loss_ds = result["diff_state_loss"]

            # Load validation losses for each task
            for task in TASKS:
                transfer_type = (
                    f"{transfer}_pt_{task}_transfer" if transfer != "scratch" else task
                )
                ckpt_dir = OTHELLO_CKPT_DIR / data_type / model / transfer_type
                val_loss = np.load(ckpt_dir / "best_val_loss.npy")
                val_miscl_rate = np.load(ckpt_dir / "best_val_miscl_rate.npy")

                data[task][model][transfer]["ib_ss"] = ib_ss
                data[task][model][transfer]["loss_ds"] = loss_ds
                data[task][model][transfer]["val_loss"] = val_loss
                data[task][model][transfer]["val_accuracy"] = 1 - val_miscl_rate

    return data


def main():
    data = load_data()

    # Print header row
    header = "Model-Transfer".ljust(30)
    for task in TASKS:
        header += task.ljust(35)
    logger.info(header)
    logger.info("-" * len(header))

    # Print each model's results
    for model in MODEL_TYPES:
        for transfer in TRANSFER_TYPES:
            row = f"{model}-{transfer}".ljust(30)
            for task in TASKS:
                task_data = data[task][model][transfer]
                val_loss_mean, val_loss_std = (
                    np.mean(task_data["val_loss"]),
                    np.std(task_data["val_loss"]),
                )
                val_loss_std_error = val_loss_std / np.sqrt(len(task_data["val_loss"]))
                val_acc_mean, val_acc_std = (
                    np.mean(task_data["val_accuracy"]),
                    np.std(task_data["val_accuracy"]),
                )
                val_acc_std_error = val_acc_std / np.sqrt(
                    len(task_data["val_accuracy"])
                )
                cell = (
                    f"NLL:{val_loss_mean:.3f}±{val_loss_std_error:.3f} "
                    f"ACC:{val_acc_mean:.3f}±{val_acc_std_error:.3f}"
                )
                row += cell.ljust(35)
            logger.info(row)

    logger.info("\nCorrelations:")
    logger.info("-" * len(header))

    # Calculate and print correlations
    metrics = ["NLL", "ACC"]
    for metric in metrics:
        row = f"IBRatio-{metric}".ljust(30)
        for task in TASKS:
            # Collect all values across models and transfer types
            ib_ss_values = []
            loss_ds_values = []
            val_acc_values = []
            val_loss_values = []

            for model in MODEL_TYPES:
                for transfer in TRANSFER_TYPES:
                    task_data = data[task][model][transfer]
                    ib_ss_values.append(task_data["ib_ss"])
                    loss_ds_values.append(task_data["loss_ds"])
                    val_acc_values.append(np.mean(task_data["val_accuracy"]))
                    val_loss_values.append(np.mean(task_data["val_loss"]))

            target_values = val_acc_values if metric == "ACC" else val_loss_values
            ib_ratio = [
                ss / (1 - ds) for ss, ds in zip(ib_ss_values, loss_ds_values)
            ]
            corr = np.corrcoef(ib_ratio, target_values)[0, 1]
            cell = f"{corr:.3f}".ljust(35)
            row += cell
        logger.info(row)


if __name__ == "__main__":
    main()
