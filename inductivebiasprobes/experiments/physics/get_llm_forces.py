import os
import numpy as np
import yaml
import json
import re
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic 
import google.generativeai as genai
from worldmodeltransfer.paths import (
    PHYSICS_DATA_DIR,
    PHYSICS_CONFIG_DIR,
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# model = "o3"
# model = "claude-sonnet-4-20250514" 
model = "gemini-2.5-pro"
model_to_provider = {
    "o1": "openai",
    "o3": "openai",
    "claude-sonnet-4-20250514": "anthropic",
    "gemini-2.5-pro": "google",
}

# Parse the predictions from the response
def parse_predictions(response_text):
    """Extract predictions from the LLM response."""
    # Look for the ANSWER: section
    answer_match = re.search(r'ANSWER:\s*({.*})', response_text, re.DOTALL)
    if not answer_match:
        print("Could not find ANSWER section in response")
        return None
    answer_text = answer_match.group(1)
    try:
        # Parse the dictionary
        predictions = eval(answer_text)  # Be careful with eval in production
        return predictions
    except Exception as e:
        print(f"Error parsing predictions: {e}")
        return None
        

def undiscretize_data(data, num_bins, min_value, max_value):
    lo, hi = min_value, max_value
    width = (hi - lo) / num_bins
    # Calculate the midpoint of each bin
    undiscretized_values = lo + (data + 0.5) * width
    return undiscretized_values


def build_sequence_string(traj, masked_inds, true_force):
    """
    Build a multiline prompt of the form
    'Timestep: i, Coordinates: traj[i], Label: ...'
    Parameters
    ----------
    traj : np.ndarray
        Coordinates for each timestep.  Supports either shape (T,) for 1-D
        data or (D, T) / (T, D) for multi-dimensional coordinates.
    masked_inds : np.ndarray or list
        1-D array/iterable of the indices whose labels are masked.
    true_force : np.ndarray
        True force values for each timestep (same indexing as `traj`).
    Returns
    -------
    str
        One string with `num_timesteps` lines.
    """
    masked_set = set(masked_inds)
    timestep_axis = 0
    num_timesteps = traj.shape[timestep_axis]
    lines = []
    for i in range(num_timesteps):
        coords = f"({traj[i][0]:.2f}, {traj[i][1]:.2f})"
        label_val = (
            "Unk" if i in masked_set
            else f"{true_force[i].item()}"
        )
        lines.append(
            f"Timestep: {i}, Coordinates: {coords}, Outcome: {label_val}"
        )
    return "\n".join(lines)

with (PHYSICS_CONFIG_DIR / ("force_magnitude_config.yaml")).open("r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

obs = np.load(PHYSICS_DATA_DIR / "obs_two_body_train.npy")
true_state = np.load(PHYSICS_DATA_DIR / "state_two_body_train.npy")
true_force_magnitudes = np.load(PHYSICS_DATA_DIR / "force_magnitude_two_body_train.npy")

masked_force_magnitudes = np.load(PHYSICS_DATA_DIR / "force_magnitude_two_body_train.npy")
max_sequence_length = 450  
magnitude_unmasked_per_sequence = 10
force_magnitude_mask_id = config["mask_id"]

rs_mask = np.random.RandomState(0)
for i in range(masked_force_magnitudes.shape[0]):
    odd_indices = np.arange(1, max_sequence_length, 2)
    unmasked_sample_inds = rs_mask.choice(odd_indices, size=magnitude_unmasked_per_sequence, replace=False)
    masked_sample_inds = np.setdiff1d(np.arange(masked_force_magnitudes.shape[1]), unmasked_sample_inds)
    masked_force_magnitudes[i, masked_sample_inds] = force_magnitude_mask_id

num_planets = 10
num_states = true_state.shape[-1]

for planet_idx in range(num_planets):
    sun_location = [0., 0.]
    traj = undiscretize_data(obs[:, :(max_sequence_length * 2), :][planet_idx, 1::2, :], config["input_vocab_size"] - 3, -50, 50)
    masked_force = masked_force_magnitudes[:, :(max_sequence_length * 2), :][planet_idx, 1::2, :]
    true_force = true_force_magnitudes[:, :(max_sequence_length * 2), :][planet_idx, 1::2, :]
    true_state_planet = true_state[:, :(max_sequence_length * 2), :][planet_idx, 1::2, :]
    masked_inds = np.unique(np.where(np.isinf(masked_force))[0])
    unmasked_inds = np.unique(np.where(~np.isinf(masked_force))[0])
    masked_force[unmasked_inds]
    traj[unmasked_inds]
    sequence_string = build_sequence_string(traj, masked_inds, true_force)
    prompt = f"""
    You are a physics expert. You are given a sequence of coordinates and outcomes.
    The coordinates are the positions of a planet in a 2-body solar system. 
    The planet is orbiting the sun.
    The sun is at the origin.

    Here is a sequence of observations. Some of them are unknown. 
    Your job is to predict the outcomes for the unknown timesteps. 

    {sequence_string}

    You can reason all you'd like, but your answer should end with "ANSWER: " followed by the predicted outcomes for all of the timesteps, even the unknown ones. 
    You should structure your predictions as a dict, where each key is a timestep and each value is the prediction.
    You should make predictions for all of the timesteps, even the ones that are known.

    Here is an example of the output format:
    ANSWER: {{
        0: 1.0e-8,
        1: 1.0e-8,
        2: 1.0e-8,
        ...
        {max_sequence_length - 1}: 1.0e-8,
    }}
    """
    os.makedirs("llm-prompts", exist_ok=True)
    with open(f"llm-prompts/{planet_idx}.txt", "w") as f:
        f.write(prompt)
    np.save(f"llm-prompts/unmasked_inds_{planet_idx}.npy", unmasked_inds)
    np.save(f"llm-prompts/true_force_{planet_idx}.npy", true_force)
    np.save(f"llm-prompts/traj_{planet_idx}.npy", traj)
    np.save(f"llm-prompts/true_state_{planet_idx}.npy", true_state_planet)


if model_to_provider[model] == "openai":
    client = OpenAI(api_key=OPENAI_API_KEY)
elif model_to_provider[model] == "anthropic":
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
elif model_to_provider[model] == "google":
    genai.configure(api_key=GOOGLE_API_KEY)
    client = genai.GenerativeModel(model)

bar = tqdm(range(num_planets))
all_preds = []
for planet_idx in bar:
    with open(f"llm-prompts/{planet_idx}.txt", "r") as f:
        prompt = f.read()
    # Only go to API if the file doesn't exist. 
    if os.path.exists(f"llm-predictions/{model}/{planet_idx}.json"):
        continue
    print(f"Sending prompt to API... (model: {model})")
    if model_to_provider[model] == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        llm_response = response.choices[0].message.content
    elif model_to_provider[model] == "anthropic":
        response = client.messages.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10_000,
        )
        llm_response = response.content[0].text
    elif model_to_provider[model] == "google":
        response = client.generate_content(prompt)
        llm_response = response.text
    parsed_preds = parse_predictions(llm_response)
    all_preds.append(parsed_preds)
    os.makedirs(f"llm-predictions/{model}", exist_ok=True)
    with open(f"llm-predictions/{model}/{planet_idx}.json", "w") as f:
        json.dump(parsed_preds, f)

