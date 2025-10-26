import cv2
import numpy as np
from typing import List, Tuple, Optional

# Simple Hough-based paper corner detector with contour fallback.

def _order_quad(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl,tr,br,bl], axis=0).astype(np.float32)


def _rect_quality(quad: np.ndarray) -> Tuple[float, float]:
    """Return (angle_dev, aspect_ratio) where angle_dev is max abs deviation from 90 deg in degrees.
    quad expected shape (4,2) ordered tl,tr,br,bl.
    """
    def angle(a, b, c):
        v1 = a - b
        v2 = c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        return ang
    tl,tr,br,bl = quad
    angs = [angle(tl,tr,br), angle(tr,br,bl), angle(br,bl,tl), angle(bl,tl,tr)]
    dev = float(max(abs(a-90.0) for a in angs))
    w = 0.5*(np.linalg.norm(tr-tl) + np.linalg.norm(br-bl))
    h = 0.5*(np.linalg.norm(tr-br) + np.linalg.norm(tl-bl))
    ar = (h / (w + 1e-8)) if w > 1e-6 else 0.0
    return dev, float(ar)


def _approx_quad_from_contour(cnt: np.ndarray) -> Optional[np.ndarray]:
    # Use convex hull, then try several epsilons; fallback to minAreaRect.
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    # Try multiple approximation strengths (from tight to loose)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(-1,2).astype(np.float32)
    # Fallback: rotated rectangle
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)  # returns 4x2 float32
    return box.astype(np.float32)


def detect_paper_corners(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns 4x2 float32 array of corners ordered [tl, tr, br, bl] or None.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 1000 / max(h, w)
    if scale < 1:
        small = cv2.resize(gray, (int(w*scale), int(h*scale)))
    else:
        small = gray
        scale = 1.0

    # Edge detection
    v = np.median(small)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(small, lower, upper)

    # Try Hough lines (probabilistic)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=0.4*min(small.shape), maxLineGap=25)
    if lines is not None and len(lines) >= 4:
        segs = lines[:,0,:]  # (N, 4)
        angles = np.arctan2(segs[:,3]-segs[:,1], segs[:,2]-segs[:,0])
        angles = (angles + np.pi/2) % np.pi - np.pi/2
        horiz_idx = np.where(np.abs(angles) < np.deg2rad(25))[0]
        vert_idx = np.where(np.abs(angles) > np.deg2rad(65))[0]
        def top2(idx):
            if len(idx) == 0:
                return []
            lengths = np.hypot(segs[idx,2]-segs[idx,0], segs[idx,3]-segs[idx,1])
            order = np.argsort(-lengths)
            return segs[idx][order[:2]]
        hsegs = top2(horiz_idx)
        vsegs = top2(vert_idx)
        if len(hsegs) >= 2 and len(vsegs) >= 2:
            def fit_line(seg):
                x1,y1,x2,y2 = map(float, seg)
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                return np.array([A,B,C], dtype=np.float64)
            linesA = [fit_line(s) for s in hsegs]
            linesB = [fit_line(s) for s in vsegs]
            def intersect(L1, L2):
                A1,B1,C1 = L1
                A2,B2,C2 = L2
                D = A1*B2 - A2*B1
                if np.abs(D) < 1e-8:
                    return None
                x = (B1*C2 - B2*C1)/D
                y = (C1*A2 - C2*A1)/D
                return np.array([x,y], dtype=np.float64)
            pts = []
            for Lh in linesA:
                for Lv in linesB:
                    p = intersect(Lh, Lv)
                    if p is not None:
                        pts.append(p)
            if len(pts) >= 4:
                pts = np.array(pts) / scale
                ordered = _order_quad(pts[:4])
                # Optional: refine corners on full-res
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
                pts_in = ordered.astype(np.float32).reshape(-1,1,2)
                corners_ref = cv2.cornerSubPix(gray, pts_in, (5,5), (-1,-1), term)
                return corners_ref.reshape(-1,2).astype(np.float32)

    # Fallback: robust contour-based quadrilateral detection
    # 1) Light normalization and denoise on downscaled image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    norm = clahe.apply(small)
    blurred = cv2.bilateralFilter(norm, d=7, sigmaColor=50, sigmaSpace=7)

    # 2) Threshold: combine adaptive + Otsu via AND to be conservative
    th_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th_adap = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)
    th = cv2.bitwise_and(th_otsu, th_adap)

    # Make paper foreground (white) for contour search
    if th.mean() < 127:
        th = 255 - th

    # 3) Morphology to close gaps, then open small noise
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    hS, wS = small.shape
    img_area = hS * wS

    best_quad = None
    best_score = None

    for cnt in contours[:7]:
        area = cv2.contourArea(cnt)
        if area < 0.15 * img_area:
            break
        quad = _approx_quad_from_contour(cnt)
        quad = quad.astype(np.float32)
        # Order and score
        quad_ord = _order_quad(quad)
        dev, ar = _rect_quality(quad_ord)
        # Accept wide AR because of perspective; prefer near A4 (1.3-1.5)
        ar_penalty = min(abs(ar-1.414), 0.8)  # capped
        score = dev + 20 * ar_penalty  # smaller is better
        if best_score is None or score < best_score:
            best_score = score
            best_quad = quad_ord

    if best_quad is None:
        return None

    # Map to original scale
    quad_full = best_quad / scale

    # Refine corners on full-res image for subpixel accuracy
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    pts_in = quad_full.astype(np.float32).reshape(-1,1,2)
    corners_ref = cv2.cornerSubPix(gray, pts_in, (7,7), (-1,-1), term)

    return corners_ref.reshape(-1,2).astype(np.float32)


def warp_to_a4(bgr: np.ndarray, corners: np.ndarray, dpi: int = 150) -> np.ndarray:
    """Perspective warp to A4 ratio canvas. Returns warped BGR image."""
    tl,tr,br,bl = corners
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    width = int(max(widthA, widthB))
    height = int(max(heightA, heightB))
    # Adjust to A4 ratio (h/w ~ 1.414)
    target_h = int(max(height, width*1.414))
    target_w = int(max(width, target_h/1.414))
    dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (target_w, target_h))
    return warped
