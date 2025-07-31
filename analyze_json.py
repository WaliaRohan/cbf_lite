import json
import sys
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def weighted_average(metrics_list):
    total_weight = 0
    cumulative = defaultdict(float)

    for entry in metrics_list:
        weight = entry['end_idx'] - entry['start_idx'] + 1
        total_weight += weight
        for key, value in entry['avg_metrics'].items():
            cumulative[key] += value * weight

    return {key: val / total_weight for key, val in cumulative.items()}

def main(file_paths, output_path):
    all_data = [load_json(path) for path in file_paths]

    # Validate consistency
    base_sim_info = {
        "sim_params": all_data[0]["sim_params"],
        "sensor_params": all_data[0]["sensor_params"],
        "control_params": all_data[0]["control_params"],
        "belief_cbf_params": all_data[0]["belief_cbf_params"]
    }

    for i, d in enumerate(all_data[1:], start=2):
        for section in base_sim_info:
            if d[section] != base_sim_info[section]:
                raise ValueError(f"{section} differs in file #{i}")

    # Compute weighted average
    avg = weighted_average(all_data)

    # Construct output
    output = {
        "start_idx": min(d["start_idx"] for d in all_data),
        "end_idx": max(d["end_idx"] for d in all_data),
        **base_sim_info,
        "avg_metrics": avg
    }

    print(json.dumps(output, indent=2))

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved aggregated results to {output_path}")

if __name__ == "__main__":
    file_paths = ["/home/rwalia/Projects/automata_lab/cbf_lite/Results/EKF_slow_meas_2/run_EKF_slow_meas_2_1_250_2025-07-15_10-29-13.json",
                  "/home/rwalia/Projects/automata_lab/cbf_lite/Results/EKF_slow_meas_2/run_EKF_slow_2_251_500_2025-07-18_10-13-26.json"
                  ]
    
    output_path = "/home/rwalia/Projects/automata_lab/cbf_lite/Results/EKF_slow_meas_2/EKF_slow_meas_2_1_500.json"
    main(file_paths, output_path)