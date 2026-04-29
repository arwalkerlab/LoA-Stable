import argparse
import json
import multiprocessing as mp
import os
import sys

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
    from pdf2image import convert_from_path
    from PIL import Image
    import numpy as np
except Exception as err:
    print(f"Required DECIMER segmentation packages not available: {err}", file=sys.stderr)
    sys.exit(1)


def _shape_to_bbox(box, h, w):
    if box is None:
        return None, None
    try:
        if len(box) < 4:
            return None, None
    except TypeError:
        return None, None
    y0, x0, y1, x1 = [int(x) for x in box]
    return [y0, x0, y1, x1], [y0 / h, x0 / w, y1 / h, x1 / w]


def _segment_array(arr):
    if HAS_INTERNAL_SEGMENT_API:
        masks, _, _ = get_mrcnn_results(arr)
        segments, boxes = apply_masks(arr, masks)
        if len(segments) > 0:
            segments, boxes = sort_segments_bboxes(segments, boxes)
        return segments, boxes
    segments = segment_chemical_structures(arr, expand=True)
    return segments, [None] * len(segments)


def _write_segment_image(output_dir, segment_index, segment_image):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"segment_{segment_index:03d}.png")
    Image.fromarray(segment_image).save(out_path, format="PNG")
    return out_path


def _segment_file(path, output_dir):
    if HAS_INTERNAL_SEGMENT_API:
        arr = np.array(Image.open(path).convert("RGB"))
        segments, boxes = _segment_array(arr)
        h, w = arr.shape[:2]
    else:
        segments = segment_chemical_structures_from_file(path, expand=True)
        boxes = [None] * len(segments)
        arr = np.array(Image.open(path).convert("RGB"))
        h, w = arr.shape[:2]

    out = []
    for seg_idx, (seg, box) in enumerate(zip(segments, boxes), start=1):
        bbox, bbox_relative = _shape_to_bbox(box, h, w)
        seg_path = _write_segment_image(output_dir, seg_idx, seg)
        out.append(
            {
                "segment_index": seg_idx,
                "bbox": bbox,
                "bbox_relative": bbox_relative,
                "segment_path": seg_path,
            }
        )
    return out


def run_segmentation(path, output_dir):
    results = []
    if path.lower().endswith(".pdf"):
        pages = convert_from_path(path, 300)
        for page_idx, page in enumerate(pages, start=1):
            page_dir = os.path.join(output_dir, f"page_{page_idx:03d}")
            arr = np.array(page)
            segments, boxes = _segment_array(arr)
            h, w = arr.shape[:2]
            for seg_idx, (seg, box) in enumerate(zip(segments, boxes), start=1):
                bbox, bbox_relative = _shape_to_bbox(box, h, w)
                seg_path = _write_segment_image(page_dir, seg_idx, seg)
                results.append(
                    {
                        "page": page_idx,
                        "segment_index": seg_idx,
                        "bbox": bbox,
                        "bbox_relative": bbox_relative,
                        "segment_path": seg_path,
                    }
                )
    else:
        results = _segment_file(path, output_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DECIMER image segmentation and save crops to disk.")
    parser.add_argument("input_path", help="Path to input image or PDF")
    parser.add_argument("--output-dir", required=True, help="Directory where segmented PNG crops are written.")
    parser.add_argument("--metadata-out", required=True, help="JSON file path to write segmentation metadata.")
    args = parser.parse_args()

    data = run_segmentation(args.input_path, args.output_dir)
    with open(args.metadata_out, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} segments")
