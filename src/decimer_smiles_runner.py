import argparse
import json
import multiprocessing as mp
import sys

mp.set_start_method("spawn", force=True)

try:
    from DECIMER import predict_SMILES
except Exception as err:
    print(f"Required DECIMER SMILES packages not available: {err}", file=sys.stderr)
    sys.exit(1)


def run_smiles_prediction(segment_paths_json):
    with open(segment_paths_json, "r") as f:
        segment_paths = json.load(f)

    results = []
    for segment_path in segment_paths:
        try:
            smi = predict_SMILES(segment_path)
            results.append(smi if smi else None)
        except Exception as err:
            print(f"SMILES prediction failed: {err}", file=sys.stderr)
            results.append(None)
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SMILES from DECIMER segment images.")
    parser.add_argument("--segment-paths-json", required=True, help="JSON file with list of segment image file paths.")
    args = parser.parse_args()
    run_smiles_prediction(args.segment_paths_json)
