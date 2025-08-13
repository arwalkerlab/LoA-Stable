import json
import sys
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

try:
    from decimer_segmentation import (
        segment_chemical_structures_from_file,
        get_mrcnn_results,
        apply_masks,
        sort_segments_bboxes,
    )
    from DECIMER import predict_SMILES
    from pdf2image import convert_from_path
    import numpy as np
except Exception as err:
    print(f"Required DECIMER packages not available: {err}", file=sys.stderr)
    sys.exit(1)


def run_decimer(path: str):
    """Run DECIMER on a file and return SMILES with optional location info."""
    results = []

    if path.lower().endswith(".pdf"):
        pages = convert_from_path(path, 300)
        for idx, page in enumerate(pages, start=1):
            arr = np.array(page)
            masks, bboxes, _ = get_mrcnn_results(arr)
            segments, boxes = apply_masks(arr, masks)
            if len(segments) > 0:
                segments, boxes = sort_segments_bboxes(segments, boxes)
            for seg, box in zip(segments, boxes):
                try:
                    smi = predict_SMILES(seg)
                    if smi:
                        y0, x0, y1, x1 = [int(x) for x in box]
                        results.append({
                            "smiles": smi,
                            "page": idx,
                            "bbox": [y0, x0, y1, x1],
                        })
                except Exception as e:
                    print(f"SMILES prediction failed: {e}", file=sys.stderr)
    else:
        segments = segment_chemical_structures_from_file(path)
        for seg in segments:
            try:
                smi = predict_SMILES(seg)
                if smi:
                    results.append({"smiles": smi})
            except Exception as e:
                print(f"SMILES prediction failed: {e}", file=sys.stderr)

    print(json.dumps(results))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: decimer_runner.py <input_path>", file=sys.stderr)
        sys.exit(1)
    run_decimer(sys.argv[1])

