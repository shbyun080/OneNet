import argparse
import csv
import os
from collections import defaultdict

from evaluate_network import evaluate_network

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()

skips = ("LUNG", "SPLEEN", "OLD", "SKIP", "UNET")
keyword = "BRAIN"

data = defaultdict(lambda: [])
for folder in os.listdir(args.folder):
    # if any((s in folder for s in skips)):
    if not any((s in folder for s in keyword)) or "ONENET_SPATIAL" in folder or "UNET" in folder:
        continue
    try:
        results = evaluate_network(os.path.join(args.folder, folder, "config.yaml"), args.cuda)
        data["name"].append(folder)
        for metric, value in results.items():
            data[metric].append(value)
    except Exception as e:
        print(f"Skipping {folder}, {e}")

csv_data = []
for key, value in data.items():
    csv_data.append([key])
    for v in value:
        csv_data[-1].append(v)
csv_data = list(zip(*csv_data))

with open(os.path.join(args.folder, "results_summary.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    for row in csv_data:
        writer.writerow(row)
