import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()


fig, axs = plt.subplots(3, 5, figsize=(15, 8))

for model_ind, model in enumerate(["o3", "claude-sonnet-4-20250514", "gemini-2.5-pro"]):
  for index in range(5):
    ax = axs[model_ind, index]
    preds = json.load(open(f"llm-predictions/{model}/{index}.json"))
    true_force = np.load(f"llm-prompts/true_force_{index}.npy")
    xs = np.array([int(k) for k in preds.keys()])
    ax.plot(xs, preds.values(), label="LLM", marker='o', markersize=1)
    ax.plot(xs, true_force, label="True", marker="o", markersize=1)
    if model_ind != 2:
        ax.set_xticklabels([])
    ax.set_xlabel("Timestep")
    if index == 0:
        ax.set_ylabel("Force Magnitude")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax.set_title(f"Solar system {index+1}")
    ax.legend(loc="upper left")

fig.tight_layout()
plt.subplots_adjust(hspace=0.8)

model_names = ["o3", "Claude Sonnet 4", "Gemini 2.5 Pro"]
for model_ind, title in enumerate(model_names):
    middle_ax = axs[model_ind, 2]  # Index 2 is the middle (0,1,2,3,4)
    fig.text(0.5, middle_ax.get_position().y1 + 0.05, f"Model: {title}", 
             ha='center', va='bottom', fontsize=16)

fig_dir = "figs"
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, 'llm_magnitude_preds.pdf'), dpi=300, bbox_inches='tight')
plt.close()