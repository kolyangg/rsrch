#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def create_face_mask(in_path: str, out_path: str):
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"could not open '{in_path}'")
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    fa = FaceAnalysis(allowed_modules=['detection'])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    faces = fa.get(img)
    if not faces:
        cv2.imwrite(out_path, mask)
        return

    # pick the face with largest bbox area
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = largest.bbox.astype(int)
    # optional padding (10%)
    pw, ph = int((x2-x1)*0.1), int((y2-y1)*0.1)
    x1, y1 = max(0, x1-pw), max(0, y1-ph)
    x2, y2 = min(w, x2+pw), min(h, y2+ph)

    cx, cy = (x1+x2)//2, (y1+y2)//2
    axes = ((x2-x1)//2, (y2-y1)//2)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    cv2.imwrite(out_path, mask)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]}  input.jpg  mask.png")
        sys.exit(1)
    create_face_mask(sys.argv[1], sys.argv[2])
