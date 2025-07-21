import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from inductivebiasprobes.paths import GRIDWORLD_EXT_DIR


def main():
    sns.set_theme(style="darkgrid")
    
    model_types = ["rnn", "lstm", "mamba", "mamba2", "gpt"]
    model_names = {
        "rnn": "RNN",
        "lstm": "LSTM",
        "mamba": "Mamba",
        "mamba2": "Mamba-2",
        "gpt": "Transformer",
    }
    num_states_list = [2, 3, 4, 5]
    
    data = []
    
    for model_type in model_types:
        for num_states in num_states_list:
            pretrained = "next_token"
            white_noise_dataset_size = 100
            max_iters = 100

            ext_curr_dir = (
                GRIDWORLD_EXT_DIR
                / f"{num_states}-states"
                / "white_noise"
                / model_type
                / f"pt_{pretrained}"
                / f"{white_noise_dataset_size}_examples"
                / f"{max_iters}_iters"
            )
            ib_file = ext_curr_dir / "ib.json"

            if ib_file.exists():
                with open(ib_file, "r") as f:
                    ib_results = json.load(f)

                r_ib = ib_results["same_state_ib"]
                # D-IB is not explicitly in the data, but the plot shows a score.
                # "diff_state_loss" is a loss, so we convert it to a score.
                # Assuming D-IB = 1 - diff_state_loss to match plot style.
                d_ib = 1 - ib_results["diff_state_loss"]

                data.append(
                    {
                        "model": model_names[model_type],
                        "num_states": num_states,
                        "r_ib": r_ib,
                        "d_ib": d_ib,
                    }
                )
            else:
                print(f"Data file not found, skipping: {ib_file}")

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot R-IB
    sns.lineplot(
        ax=axes[0],
        x="num_states",
        y="r_ib",
        hue="model",
        data=df,
        marker="o",
        markersize=8,
    )
    axes[0].set_title("R-IB as a function of states")
    axes[0].set_xlabel("Number of states")
    axes[0].set_ylabel("R-IB")
    axes[0].legend(title="")

    # Plot D-IB
    sns.lineplot(
        ax=axes[1],
        x="num_states",
        y="d_ib",
        hue="model",
        data=df,
        marker="o",
        markersize=8,
    )
    axes[1].set_title("D-IB as a function of states")
    axes[1].set_xlabel("Number of states")
    axes[1].set_ylabel("D-IB")
    axes[1].legend(title="")

    # Improve layout and save the figure
    plt.tight_layout()
    plt.savefig("ib_vs_num_states.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
