import argparse
from pathlib import Path
import cv2
import numpy as np

from HW_1.src.model import load_or_train_default
from .paper_detect import detect_paper_corners, warp_to_a4
from .segments import binarize, extract_components, crop_and_resize


def process_image(path: Path, model, model_path: Path) -> str:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return f"{path.name}: [unreadable]"
    corners = detect_paper_corners(bgr)
    if corners is None:
        return f"{path.name}: [paper not found]"
    warped = warp_to_a4(bgr, corners)
    bin_img = binarize(warped)
    boxes = extract_components(bin_img)
    if not boxes:
        return f"{path.name}: []"
    if model is None:
        return f"{path.name}: components={len(boxes)} (classification disabled)"
    patches = np.stack([crop_and_resize(warped, b, 28) for b in boxes], axis=0)
    preds = predict_digits(model, patches)
    digits = ''.join(map(str, preds.tolist()))
    return f"{path.name}: {digits}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', type=str, required=True, help='Directory with input images')
    ap.add_argument('--model', type=str, default='HW_1/model.h5')
    args = ap.parse_args()

    images = []
    for ext in ('*.jpg','*.png','*.jpeg','*.JPG','*.PNG'):
        images.extend(sorted(Path(args.images_dir).glob(ext)))
    if not images:
        print('No images found')
        return

    model_path = Path(args.model)
    model = None
    try:
        model = load_or_train_default(str(model_path))
    except Exception as e:
        print('TensorFlow/Keras not available. Proceeding without classification.')

    for p in images:
        print(process_image(p, model, model_path))


if __name__ == '__main__':
    main()
