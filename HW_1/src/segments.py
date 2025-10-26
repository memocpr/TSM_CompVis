import cv2
import numpy as np
from typing import List, Tuple


def binarize(warp_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    # Remove uneven illumination using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    norm = cv2.normalize(gray - bg, None, 0, 255, cv2.NORM_MINMAX)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(norm.astype(np.uint8), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    # Small noise removal
    th = cv2.medianBlur(th, 3)
    return th


def extract_components(bin_img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    # Connected components with stats
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, 8)
    boxes = []
    H,W = bin_img.shape
    for i in range(1, num):  # skip background
        x,y,w,h,area = stats[i,0], stats[i,1], stats[i,2], stats[i,3], stats[i,4]
        if area < 50:
            continue
        # Skip very large regions (likely page edges or artifacts)
        if w*h > 0.25*H*W:
            continue
        # Pad and square the box
        pad = int(0.1*max(w,h))
        x0 = max(0, x-pad)
        y0 = max(0, y-pad)
        x1 = min(W, x+w+pad)
        y1 = min(H, y+h+pad)
        boxes.append((x0,y0,x1-x0,y1-y0))
    # Sort by top-to-bottom
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def crop_and_resize(gray_or_bgr: np.ndarray, box: Tuple[int,int,int,int], size: int=28) -> np.ndarray:
    if len(gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    x,y,w,h = box
    crop = gray[y:y+h, x:x+w]
    # Keep aspect and pad to square
    h0, w0 = crop.shape
    m = max(h0, w0)
    canvas = np.zeros((m,m), dtype=np.uint8)
    y_off = (m - h0)//2
    x_off = (m - w0)//2
    canvas[y_off:y_off+h0, x_off:x_off+w0] = crop
    resized = cv2.resize(canvas, (size,size), interpolation=cv2.INTER_AREA)
    return resized

