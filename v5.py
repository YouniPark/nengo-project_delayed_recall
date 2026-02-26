import nengo
import nengo_spa as spa
import numpy as np
import random
import math
import time
import os
import csv
import matplotlib.pyplot as plt
from english_words import get_english_words_set
from nengo.exceptions import ValidationError
import re
from dataclasses import dataclass

# -----------------------------------------
# Utility functions
# -----------------------------------------

def uppercase(word):
    key = word.upper()
    key = re.sub(r'[^A-Z0-9_]', '_', key)
    if key and key[0].isdigit():
        key = "X" + key
    return key


def shuffle_no_repeat(categories_dict):
    sequence = []
    category_order = [cat for cat in categories_dict.keys() for _ in range(4)]
    while True:
        random.shuffle(category_order)
        if all(category_order[i] != category_order[i + 1] for i in range(len(category_order) - 1)):
            break
    categories_copy = {k: v.copy() for k, v in categories_dict.items()}
    for cat in category_order:
        sequence.append(categories_copy[cat].pop())
    return sequence


# -----------------------------------------
# Experiment configuration
# -----------------------------------------

@dataclass
class ExperimentConfig:
    dim: int = 512
    n_neurons: int = 500
    scale: float = 3.33
    dt: float = 0.001
    n_trials: int = 10
    presentation_time: float = 14.0
    n_words: int = 16
    word_duration: float = 0.875
    extra_time: float = 30.0   # large ONCE


# -----------------------------------------
# Main experiment class
# -----------------------------------------

class SPASequentialRecallExperiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.vocab = spa.Vocabulary(dimensions=cfg.dim)
        self._build_vocab()

    # -------------------------------------
    # Vocabulary
    # -------------------------------------

    def _build_vocab(self):
        categories = {
            "fruits": ["APPLE", "ORANGE", "BANANA", "GRAPE"],
            "clothing": ["SHIRT", "PANTS", "JACKET", "HAT"],
            "flowers": ["ROSE", "TULIP", "DAISY", "LILY"],
            "herbs": ["BASIL", "MINT", "ROSEMARY", "THYME"]
        }

        self.stim_words = shuffle_no_repeat(categories)

        english_words = list(get_english_words_set(['web2'], lower=False))
        random.shuffle(english_words)
        english_words = [uppercase(w) for w in english_words[:2000]]

        order_words = [
            'ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT',
            'NINE','TEN','ELEVEN','TWELVE','THIRTEEN','FOURTEEN',
            'FIFTEEN','SIXTEEN'
        ]

        all_words = list(self.stim_words) + english_words + order_words

        for w in all_words:
            try:
                vec = np.random.randn(self.cfg.dim)
                vec /= np.linalg.norm(vec)
                self.vocab.add(w, vec)
            except ValidationError:
                pass

    # -------------------------------------
    # Run single trial (ONE simulation)
    # -------------------------------------

    def run_single_trial(self):
        cfg = self.cfg
        dim = cfg.dim

        model = nengo.Network()
        with model:
            radius = 1 / math.sqrt(dim)

            # ONLY word encoding is neural
            ensem_word = nengo.networks.EnsembleArray(
                n_neurons=cfg.n_neurons,
                n_ensembles=dim,
                radius=radius
            )

            ensem_order = nengo.networks.EnsembleArray(
                n_neurons=cfg.n_neurons,
                n_ensembles=dim,
                radius=radius,
                neuron_type=nengo.Direct()
            )

            ensem_conv = nengo.networks.EnsembleArray(
                n_neurons=cfg.n_neurons,
                n_ensembles=dim,
                radius=radius
            )

            cconv = nengo.networks.CircularConvolution(cfg.n_neurons, dim)
            for ens in cconv.all_ensembles:
                ens.neuron_type = nengo.Direct()

            def word_stim(t):
                idx = int(t // cfg.word_duration)
                if 0 <= idx < cfg.n_words:
                    return self.vocab[self.stim_words[idx]].v
                return np.zeros(dim)

            def order_stim(t):
                idx = int(t // cfg.word_duration)
                if 0 <= idx < cfg.n_words:
                    return self.vocab[list(self.vocab.keys())[idx]].v
                return np.zeros(dim)

            word_node = nengo.Node(word_stim)
            order_node = nengo.Node(order_stim)

            tau = 0.3
            nengo.Connection(word_node, ensem_word.input, synapse=tau,
                             transform=tau * cfg.scale)
            nengo.Connection(order_node, ensem_order.input)

            nengo.Connection(ensem_word.output, cconv.input_a)
            nengo.Connection(ensem_order.output, cconv.input_b)
            nengo.Connection(cconv.output, ensem_conv.input)

            for ens in ensem_conv.all_ensembles:
                nengo.Connection(ens, ens, synapse=tau)

            probe = nengo.Probe(ensem_conv.output, synapse=0.2)

        sim_time = cfg.presentation_time + cfg.extra_time
        start = time.time()

        with nengo.Simulator(model, dt=cfg.dt) as sim:
            sim.run(sim_time)

        return {
            "probe": sim.data[probe],
            "trial_time": time.time() - start
        }

    # -------------------------------------
    # Decode at arbitrary delay
    # -------------------------------------

    def decode_at_delay(self, probe_data, delay):
        cfg = self.cfg
        final_word_time = cfg.n_words * cfg.word_duration
        t_abs = final_word_time + delay
        idx = min(int(t_abs / cfg.dt), len(probe_data) - 1)

        encoded_sp = spa.SemanticPointer(probe_data[idx])

        decoded = []
        for i in range(cfg.n_words):
            order_sp = self.vocab[list(self.vocab.keys())[i]]
            unbound = encoded_sp * ~order_sp
            sims = spa.similarity(
                unbound.v.reshape(1, -1),
                self.vocab,
                normalize=True
            )
            decoded.append(list(self.vocab.keys())[sims[0].argmax()])

        return sum(d == t for d, t in zip(decoded, self.stim_words)) / cfg.n_words


# -----------------------------------------
# Run experiment (multi-delay, no re-run)
# -----------------------------------------

def run_experiment(cfg, delays):
    acc_by_delay = {d: [] for d in delays}
    total_time = 0.0

    for _ in range(cfg.n_trials):
        exp = SPASequentialRecallExperiment(cfg) # new initiation per trial
        result = exp.run_single_trial()
        total_time += result["trial_time"]

        for d in delays:
            acc = exp.decode_at_delay(result["probe"], d)
            acc_by_delay[d].append(acc)
            print(f"Trial {_+1}, Delay {d}s: Accuracy = {acc*100:.2f}%")

    stats = {}
    for d, accs in acc_by_delay.items():
        arr = np.array(accs)
        stats[d] = {
            "mean": arr.mean(),
            "sd": arr.std(ddof=1),
            "ci95": 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
        }

    return stats, total_time

def print_cfg(cfg: ExperimentConfig):

    print("Current ExperimentConfig:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")

def exp_increment_until_change(
        base_cfg: ExperimentConfig,
        param_name: str,
        start: float,
        step: float,
        min_value: float,
        delays,
        change_threshold=0.05,  # 5% change triggers stop
        max_iters=50,
        test_values=None
    ):
    """
    Sweep parameter up and down from start.
    Stop early if accuracy changes significantly at the last delay.
    Save results to both txt and CSV.
    Returns all tested values, full delay stats, and % change at last delay.
    """

    results = {}

    # File paths
    base_txt = f"experiment_results_{param_name}.txt"
    base_csv = f"experiment_results_{param_name}.csv"

    version = 0
    results_file_txt = base_txt
    results_file_csv = base_csv
    while os.path.exists(results_file_txt) or os.path.exists(results_file_csv):
        version += 1
        results_file_txt = f"{base_txt.rstrip('.txt')}_v{version}.txt"
        results_file_csv = f"{base_csv.rstrip('.csv')}_v{version}.csv"

    # --- TXT Header ---
    with open(results_file_txt, "w") as f:
        f.write("===== PARAMETER SWEEP =====\n")
        f.write(f"Parameter: {param_name}\n")
        f.write(f"Start value: {start}\n")
        f.write(f"Step size: {step}\n")
        f.write(f"Minimum value: {min_value}\n")
        f.write(f"Change threshold: {change_threshold*100:.1f}%\n\n")

    # --- CSV Header ---
    with open(results_file_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([f"{param_name}", "Delay (s)", "Mean Accuracy (%)", "± 95% CI (%)"])

    # -----------------------------------
    # Function to run experiment at a value
    # -----------------------------------
    def run_at_value(value):
        cfg_kwargs = base_cfg.__dict__.copy()
        cfg_kwargs[param_name] = type(start)(value)
        cfg = ExperimentConfig(**cfg_kwargs)

        print("\n----------------------------------------")
        print(f"Testing {param_name} = {value}")
        print_cfg(cfg)
        print("----------------------------------------")
        
        stats, total_time = run_experiment(cfg, delays)

        acc_last = stats[delays[-1]]["mean"]
        results[value] = {
            "stats": stats,
            "total_time": total_time,
            "acc_last": acc_last
        }

        print("\nAccuracy vs delay:")
        for d, s in stats.items():
            print(f"{d:>3}s → {s['mean']*100:.2f}% ± {s['ci95']*100:.2f}%")

        # --- Append to TXT ---
        with open(results_file_txt, "a") as f:
            f.write(f"Parameter value: {value}\n")
            for d, s in stats.items():
                f.write(
                    f"  Delay {d}s: "
                    f"{s['mean']*100:.2f}% ± {s['ci95']*100:.2f}%\n"
                )
            f.write(
                f"  Total computation time: {total_time/60:.2f} minutes\n\n"
            )

        # --- Append to CSV ---
        with open(results_file_csv, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            for d, s in stats.items():
                writer.writerow([value, d, s['mean']*100, s['ci95']*100])

        print(f"Total computation time: {total_time/60:.2f} minutes")
        return acc_last

    # --- 1. Always test the start value ---
    acc_start = run_at_value(start)

    # --- 2. Prepare sweep values for n_neurons ---
    if test_values is not None:
        lower_values = [v for v in test_values if v < start and v >= min_value][::-1]
        upper_values = [v for v in test_values if v > start]

        # Remove start value after casting
        lower_values = [v for v in lower_values if type(start)(v) != start]
        upper_values = [v for v in upper_values if type(start)(v) != start]

    # --- 3. Sweep ---
    for direction in (-1, 1):
        for i in range(1, max_iters + 1):
            if test_values is not None:
                values_list = lower_values if direction == -1 else upper_values
                if i-1 >= len(values_list):
                    break
                current = values_list[i-1]
            else:
                current = start + direction * i * step
                if current < min_value:
                    continue
                    
            acc_current = run_at_value(current)
            # pct_change = (acc_current - acc_start) * 100
            # print(f"  %Δ from start: {pct_change:.2f}%")

            # # Stop if change exceeds threshold
            # if abs(acc_current - acc_start) >= change_threshold:
            #     print(f"\n⚠️ Significant change detected at {param_name} = {current}")
            #     return {
            #         "trigger_value": current,
            #         "pct_change": pct_change,
            #         "results": results
            #     }   

    print("\nNo significant change detected within bounds.")
    # return {"results": results, "trigger_value": None, "pct_change": 0.0}
    return



# -----------------------------------------
# MAIN
# -----------------------------------------

def main_operator():

    cfg = ExperimentConfig( # default for other fixed values
        dim=512,
        n_neurons=500,
        dt=0.001,
        scale=3.33,
        n_trials=10,
        extra_time=30
    )

    delays = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

    # --- Increment a parameter until change ---
    # param_to_test = "n_neurons"
    # start_value = 750      # start lower than current default
    # test_values = [1, 5, 10, 15, 20, 25, 50, 100, 150, 200, 300, 
    #                1500, 2000, 2500, 3000, 4000, 5000, 7500, 9000]
    # step_size = 0          # increment size
    # min_value = 1          # avoid going too low
    # change_threshold = 0.05 # 5% change in accuracy is considered significant

    # param_to_test = "n_neurons"
    # start_value = 20 
    # test_values =  [300, 750]
    step_size = 0
    min_value = 1
    change_threshold = 0.05 # 5% change in accuracy is considered significant

    param_to_test = "scale"
    start_value = 5.66
    test_values =  [5.99, 6.33, 6.66, 6.99, 7.33, 7.66] 

    exp_increment_until_change(
        base_cfg=cfg,
        param_name=param_to_test,
        start=start_value,      # your current default
        step=step_size,        # typical meaningful change for scale
        min_value=min_value,   # avoid instability / vanishing input
        delays=delays,
        change_threshold=change_threshold,
        max_iters=10,
        test_values=test_values
    )

    # --- 4. Plot accuracy vs. parameter sweep for each delay ---
    # for delay in delays:
    #     plt.figure(figsize=(8,5))
    #     param_values = sorted(results_scale["results"].keys())
    #     accuracies = [results_scale["results"][v]["stats"][delay]["mean"]*100 for v in param_values]
    #     plt.plot(param_values, accuracies, marker="o")
    #     plt.xlabel(f"{param_to_test} value")
    #     plt.ylabel("Accuracy (%)")
    #     plt.title(f"Accuracy vs. {param_to_test} at delay {delay}s")
    #     plt.grid(True)
    #     plt.show()


# For testing

def debug():

    cfg = ExperimentConfig(
        dim=512,
        n_neurons=500,
        dt=0.001,
        scale=3.33,
        n_trials=10,
        extra_time=30
    )

    delays = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

    # --- Increment a parameter until change ---
    param_to_test = "n_neurons"
    start_value = 500      # start lower than current default
    step_size = 1.25          # increment size
    min_value = 10          # avoid going too low
    change_threshold = 0.05 # 5% change in accuracy is considered significant
    max_iters = 20

    # --- 2. Sweep ---
    for direction in (-1, 1):  # -1 for decreasing, +1 for increasing
        print(f"direction: {direction}")
        for i in range(1, max_iters + 1):
            if param_to_test == "n_neurons":
                if direction == -1:
                    # Decrease from start_value by factor^i
                    current = max(min_value, int(start_value / (step_size ** i)))
                else:
                    # Increase from start_value by factor^i
                    current = int(start_value * (step_size ** i))
            else:
                current = start_value + direction * i * 2

            print(f"Testing {param_to_test} = {current}")

    # # --- 1. Run the single experiment first ---
    # stats, total_time = run_experiment(cfg, delays)

    # print("stats:", stats)

    # print("\nAccuracy vs delay:")
    # for d, s in stats.items():
    #     print(f"{d:>3}s → {s['mean']*100:.2f}% ± {s['ci95']*100:.2f}%")
    # print(f"Total computation time: {total_time/60:.2f} minutes")
    
def find_high_accuracy():
    # Example function to find parameter values that yield high accuracy at long delays
    cfg = ExperimentConfig(
        dim=768,
        n_neurons=500,
        dt=0.001,
        scale=3.66,
        n_trials=10,
        extra_time=30
    )

    delays = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

    run_experiment(cfg, delays)

if __name__ == "__main__":

    # main_operator()
    # debug()
    find_high_accuracy()