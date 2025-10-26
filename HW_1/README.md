**HW_1 Digit Extraction Pipeline**

**Quick start**
- Python 3.10+
- pip install -r requirements.txt
- Run: python -m src.pipeline --images-dir images1

**What it does**
- Detects the paper (Canny/Hough + contour fallback), warps to A4, binarizes, and extracts connected components.
- Classification is optional: if TensorFlow/Keras is available, predicts digits; otherwise it prints components=<N> (classification disabled).


**Files**
- src/
  - pipeline.py: CLI to process a folder; detection, warp, binarize, components; optional classification.
  
  - paper_detect.py: Finds paper corners (Hough + contour fallback) and warps to A4.
  
  - segments.py: Illumination fix, adaptive threshold, connected components, crop/resize to 28×28.

  - model.py: Optional MNIST CNN load/train and prediction (uses TensorFlow if available).



