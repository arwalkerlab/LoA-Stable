import json
import sys
import multiprocessing as mp
import argparse
import base64
from io import BytesIO

mp.set_start_method("spawn", force=True)

try:
    from decimer_segmentation import (
        segment_chemical_structures,
        segment_chemical_structures_from_file,
        get_mrcnn_results,
        apply_masks,
        sort_segments_bboxes,
    )
    HAS_INTERNAL_SEGMENT_API = True
except Exception:
    from decimer_segmentation import (
        segment_chemical_structures,
        segment_chemical_structures_from_file,
    )
    HAS_INTERNAL_SEGMENT_API = False

try:
    from DECIMER import predict_SMILES
    from pdf2image import convert_from_path
    from PIL import Image
    import numpy as np
except Exception as err:
    print(f"Required DECIMER packages not available: {err}", file=sys.stderr)
    sys.exit(1)


def _encode_segment_image(segment):
    """Encode a segmented numpy image as base64 PNG."""
    with BytesIO() as buf:
        Image.fromarray(segment).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _segment_array(arr):
    """Return segmented sub-images and bounding boxes for an RGB array."""
    if HAS_INTERNAL_SEGMENT_API:
        masks, _, _ = get_mrcnn_results(arr)
        segments, boxes = apply_masks(arr, masks)
        if len(segments) > 0:
            segments, boxes = sort_segments_bboxes(segments, boxes)
        return segments, boxes
    # Public API fallback (does not expose original-page bounding boxes)
    segments = segment_chemical_structures(arr, expand=True)
    return segments, [None] * len(segments)


def _shape_to_bbox(box, h, w):
    """Build absolute and relative bbox data if a box is available."""
    if box is None:
        return None, None
    try:
        if len(box) < 4:
            return None, None
    except TypeError:
        return None, None
    y0, x0, y1, x1 = [int(x) for x in box]
    return [y0, x0, y1, x1], [y0 / h, x0 / w, y1 / h, x1 / w]


def run_decimer(path: str, mode: str = "smiles", predict_smiles: bool = True):
    """Run DECIMER on a file and return SMILES and/or segmentation metadata."""
    results = []

    if path.lower().endswith(".pdf"):
        pages = convert_from_path(path, 300)
        for idx, page in enumerate(pages, start=1):
            arr = np.array(page)
            segments, boxes = _segment_array(arr)
            h, w = arr.shape[:2]
            for seg_idx, (seg, box) in enumerate(zip(segments, boxes), start=1):
                bbox, bbox_relative = _shape_to_bbox(box, h, w)
                item = {
                    "page": idx,
                    "segment_index": seg_idx,
                    "bbox": bbox,
                    "bbox_relative": bbox_relative,
                }
                if mode == "segments":
                    item["segment_image_base64"] = _encode_segment_image(seg)
                if predict_smiles:
                    try:
                        smi = predict_SMILES(seg)
                        if smi:
                            item["smiles"] = smi
                    except Exception as e:
                        print(f"SMILES prediction failed: {e}", file=sys.stderr)
                if mode == "segments" or item.get("smiles"):
                    results.append(item)
    else:
        arr = np.array(Image.open(path).convert("RGB"))
        if HAS_INTERNAL_SEGMENT_API:
            segments, boxes = _segment_array(arr)
        else:
            segments = segment_chemical_structures_from_file(path, expand=True)
            boxes = [None] * len(segments)
        h, w = arr.shape[:2]
        for seg_idx, (seg, box) in enumerate(zip(segments, boxes), start=1):
            bbox, bbox_relative = _shape_to_bbox(box, h, w)
            item = {
                "segment_index": seg_idx,
                "bbox": bbox,
                "bbox_relative": bbox_relative,
            }
            if mode == "segments":
                item["segment_image_base64"] = _encode_segment_image(seg)
            if predict_smiles:
                try:
                    smi = predict_SMILES(seg)
                    if smi:
                        item["smiles"] = smi
                except Exception as e:
                    print(f"SMILES prediction failed: {e}", file=sys.stderr)
            if mode == "segments" or item.get("smiles"):
                results.append(item)

    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DECIMER prediction/segmentation.")
    parser.add_argument("input_path", help="Path to input image or PDF")
    parser.add_argument(
        "--mode",
        choices=["smiles", "segments"],
        default="smiles",
        help="Output mode: smiles-only or full segmentation records.",
    )
    parser.add_argument(
        "--predict-smiles",
        choices=["y", "n"],
        default="y",
        help="Whether to run DECIMER SMILES prediction on segments.",
    )
    args = parser.parse_args()
    if not args.input_path:
        print("Usage: decimer_runner.py <input_path> [--mode smiles|segments]", file=sys.stderr)
        sys.exit(1)
    run_decimer(args.input_path, mode=args.mode, predict_smiles=args.predict_smiles == "y")
