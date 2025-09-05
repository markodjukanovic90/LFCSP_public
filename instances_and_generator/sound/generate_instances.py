import os
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import random

'''
This is an instance generator that shows the applicability of solving LFCSP in audio fingerprinting. 
Following on the problem formulation from paper, here B represents the imperfect recording of song and there is a set of candidate songs As, As = {A1, ..., A8} which are candidate songs from song database -- one of them is indeed song B, in this case 03_Fugees. 
We look at energy profile of each candidate song, where energy levels are discretized to (vertical_bins) bins, so valid values are from alphabet {0, .., 9}. 
(Horizontal step refers to time dimension discretization, i.e. each horizontal value corresponds to a 1s of song.)
Then, for each candidate song we calculate energy histogram. 
After that, we calculate the common energy histogram across these 8 songs and that common histogram will basically represent M (multiset) of symbols from LFCSP formulation. 
Then, for each instance we remove the common histogram out in a random way. 
Then in addition, from each we remove (extra_removal_fraction) of symbols to make LFCSP instance harder. 
The goal is then to run LFCSP algorithm(s) and to find the most probable song. 
This is well motivated in the sound fingerprinting, i.e. searching for the most probable songs when recording is made through imperfect conditions, i.e. imperferect mobile phone sound recorder + external noise. 
So, some symbols are fully lost (extra_removal_fraction), some symbols are detected (Bs), but there might be gaps in between them due to time inconsistency, variable internet bandwidth, etc.
Also, for some symbols (energy levels) we know they appeared (common histogram M) but we do not know their exact position due to the above reasons. 
'''

# === CONFIGURATION ===
input_folder = "wavs"
horizontal_step = 1.0
vertical_bins = 10
selected_song_name = "03_Fugees"     
extra_removal_fraction = 0.0   
random_seed = 42                 
output_instance_folder = f"instances/hstep_{horizontal_step}_vstep_{vertical_bins}_sel_{selected_song_name}_rem_{extra_removal_fraction:02}_seed_{random_seed}"

os.makedirs(output_instance_folder, exist_ok=True)
random.seed(random_seed)

def compute_energy_profile(filepath, time_step):
    rate, data = wav.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    step_size = int(rate * time_step)
    return np.array([
        np.sum(data[i:i + step_size].astype(np.float32) ** 2)
        for i in range(0, len(data), step_size)
    ])

# === Load all profiles and compute histograms ===
all_profiles = []
histograms = []
bin_edges = None

for wav_file in Path(input_folder).glob("*.wav"):
    song_name = wav_file.stem
    profile = compute_energy_profile(wav_file, horizontal_step)
    all_profiles.append((song_name, profile))

    if bin_edges is None:
        _, bin_edges = np.histogram(profile, bins=vertical_bins)

    hist, _ = np.histogram(profile, bins=bin_edges)
    histograms.append((song_name, hist))

if not all_profiles:
    raise RuntimeError("No .wav files found in 'wavs/' folder.")

# === Compute common histogram ===
common_histogram = np.min([h for _, h in histograms], axis=0)
common_histogram_multiset = ''.join(str(i) * c for i, c in enumerate(common_histogram))
total_common = np.sum(common_histogram)

# === Create sequence B from selected song ===
selected_profile = None
for name, profile in all_profiles:
    if name == selected_song_name:
        selected_profile = profile
        break

if selected_profile is None:
    raise ValueError(f"Selected song '{selected_song_name}' not found.")

bin_indices = np.digitize(selected_profile, bins=bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, vertical_bins - 1)
sequence_A_sel = bin_indices.tolist()
sequence_B = sequence_A_sel.copy()

# Step 1: Remove values based on common histogram
hist_copy = common_histogram.copy()
bin_locations = {i: [] for i in range(vertical_bins)}
for idx, val in enumerate(sequence_B):
    bin_locations[val].append(idx)

for val, count in enumerate(hist_copy):
    if count > 0 and bin_locations[val]:
        count = min(count, len(bin_locations[val]))
        to_remove = random.sample(bin_locations[val], count)
        for idx in to_remove:
            sequence_B[idx] = '_'

# Step 2: Remove additional fraction from remaining
remaining_idxs = [i for i, v in enumerate(sequence_B) if v != '_']
num_extra_remove = int(len(remaining_idxs) * extra_removal_fraction)
extra_to_remove = random.sample(remaining_idxs, num_extra_remove)
for idx in extra_to_remove:
    sequence_B[idx] = '_'

# Final version of B
line3_B = ''.join(str(x) if x != '_' else '_' for x in sequence_B)
line3_B = line3_B.replace('_', '')

# === Write one file per song ===
for name, profile in all_profiles:
    bin_indices = np.digitize(profile, bins=bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, vertical_bins - 1)
    sequence_A = ''.join(str(x) for x in bin_indices.tolist())

    line1 = f"{vertical_bins} {total_common}"
    line2 = sequence_A
    line4 = common_histogram_multiset

    out_path = os.path.join(output_instance_folder, f"{name}.txt")
    with open(out_path, "w") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3_B + "\n")
        f.write(line4 + "\n")

    print(f"✅ Written: {out_path}")
