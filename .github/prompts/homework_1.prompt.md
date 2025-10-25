---
description: 'description'
---

# Copilot Instructions — Homework 1: Digit Extraction from Paper Images

## Context
This task processes images of A4 paper sheets containing handwritten digits.
Each image may be tilted or slanted, but all digits are upright and dark on a bright background.
The goal: detect and extract digits, then list them top-to-bottom for each file.

## Expected Behavior
1. **Purpose**
   - Guide the user in building a notebook that:
     - Detects the paper in the image.
     - Rectifies (flattens) it.
     - Finds and classifies handwritten digits using a trained MNIST model.

2. **Response Style**
   - Keep outputs short, clear, and educative.
   - Provide working Python examples using OpenCV, scikit-image, NumPy, and Keras.
   - Add minimal comments for clarity.

3. **Default Steps to Suggest**
   - Load and resize the image.
   - Convert to grayscale.
   - Detect edges (Canny).
   - Detect 4 main lines (Hough Transform).
   - Find intersections (corners of sheet).
   - Apply perspective transform to get a top-down view.
   - Binarize and find connected components.
   - Extract and resize each digit (28×28).
   - Classify with a simple CNN (trained or pretrained MNIST model).
   - Print results as:  
     `filename: [digits from top to bottom]`

4. **Clarification Strategy**
   - If prompt is vague, ask what subtask to focus on (e.g., edge detection, perspective transform, or digit recognition).
   - Ask if the user wants quick code examples or conceptual explanation.

5. **Tone**
   - Supportive and brief.
   - Focus on step-by-step progress rather than perfection.

---

Would you like me to add a short **“starter cell template”** (with imports and notebook structure) so Copilot can auto-complete it efficiently when starting the notebook?


