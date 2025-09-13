import os
import json
import numpy as np

def build_exclude_files(numbers):
    """Generate excluded filenames from list of numbers."""
    return [f"{i}_result.json" for i in numbers]

def compute_mean_metrics(folder_path, exclude_files=None):
    if exclude_files is None:
        exclude_files = []

    metrics_data = {}
    nan_report = {}
    file_values = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        if filename in exclude_files:
            print(f"⏭️ Skipping excluded file: {filename}")
            continue

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {filename}")
                continue

        if "metrics" not in data:
            continue

        file_metrics = {}
        bad_metrics = []

        # Try converting all metrics first
        for key, val in data["metrics"].items():
            try:
                num_val = float(val)
                if np.isnan(num_val):
                    bad_metrics.append(key)
            except (ValueError, TypeError):
                bad_metrics.append(key)
            else:
                file_metrics[key] = num_val

        # If any bad metric → skip whole file
        if bad_metrics:
            for m in bad_metrics:
                nan_report.setdefault(m, []).append(filename)
            print(f"⚠️ Skipping entire file due to NaN/invalid values: {filename}")
            continue

        # Otherwise add values
        for key, num_val in file_metrics.items():
            metrics_data.setdefault(key, []).append(num_val)
            file_values.setdefault(key, []).append((filename, num_val))

    # Compute means
    mean_metrics = {k: float(np.mean(v)) for k, v in metrics_data.items() if v}

    # Compute top 5
    top_files = {}
    for k, vals in file_values.items():
        if k.startswith("min_"):
            sorted_vals = sorted(vals, key=lambda x: x[1])[:5]
        else:
            sorted_vals = sorted(vals, key=lambda x: x[1], reverse=True)[:5]
        top_files[k] = sorted_vals

    return mean_metrics, nan_report, top_files


if __name__ == "__main__":
    folder = "./batch_results/2025-09-08_03-48-10"  # change to your folder path

    # Example: exclude reports 3, 7, and 15
    exclude_numbers = [11, 22, 26, 27, 40, 41, 44, 51, 53, 59, 62, 68, 73, 90, 91]
    exclude_files = build_exclude_files(exclude_numbers)

    mean_metrics, nan_report, top_files = compute_mean_metrics(folder, exclude_files=exclude_files)

    print("=== Mean Metrics Across Files ===")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.6f}")

    if nan_report:
        print("\n=== Files Skipped Due to NaN/Invalid Values ===")
        for k, files in nan_report.items():
            print(f"{k}: found in {files}")

    print("\n=== Top 5 Files by Metric ===")
    for k, entries in top_files.items():
        print(f"\n{k}:")
        for fname, val in entries:
            print(f"  {fname}: {val:.6f}")
