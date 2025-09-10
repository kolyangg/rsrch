# # create_mask2.py
# import sys, cv2, numpy as np, mediapipe as mp

# def highlight_face(src_path, dst_path, alpha=0.6):
#     img = cv2.imread(src_path);  h, w = img.shape[:2]
#     mesh = mp.solutions.face_mesh
#     mask = np.zeros((h, w), np.uint8)

#     with mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as fm:
#         res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     if res.multi_face_landmarks:
#         best_hull, best_area = None, 0
#         for face in res.multi_face_landmarks:
#             pts = []
#             for i, j in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
#                 for idx in (i, j):
#                     lm = face.landmark[idx]
#                     pts.append([int(lm.x * w), int(lm.y * h)])
#             hull = cv2.convexHull(np.array(pts, np.int32))
#             area = cv2.contourArea(hull)
#             if area > best_area: best_hull, best_area = hull, area
#         if best_hull is not None:
#             cv2.fillConvexPoly(mask, best_hull, 255)

#     red = np.full_like(img, (0, 0, 255))
#     m3  = cv2.merge([mask, mask, mask])
#     out = np.where(m3 > 0, (alpha * red + (1 - alpha) * img).astype(np.uint8), img)
#     cv2.imwrite(dst_path, out)

# if __name__ == "__main__":
#     highlight_face(sys.argv[1], sys.argv[2])


# highlight_face.py
import sys, cv2, numpy as np, mediapipe as mp

def highlight_face(src_path, dst_path, alpha=0.6, scale=1.05, top_scale=1.20, dilate_frac=0.04):
    """
    scale: uniform polygon growth (1.10–1.30)
    top_scale: extra vertical growth for points above center (1.00–1.20)
    dilate_frac: fraction of image size for dilation kernel (0.00–0.06)
    """
    img = cv2.imread(src_path);  h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mesh = mp.solutions.face_mesh

    with mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if res.multi_face_landmarks:
        best_hull, best_area = None, 0
        for face in res.multi_face_landmarks:
            pts = []
            for i, j in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
                for idx in (i, j):
                    lm = face.landmark[idx]
                    pts.append([int(lm.x * w), int(lm.y * h)])
            hull = cv2.convexHull(np.array(pts, np.int32))
            area = cv2.contourArea(hull)
            if area > best_area: best_hull, best_area = hull, area

        if best_hull is not None:
            hull = best_hull.reshape(-1, 2).astype(np.float32)
            cx, cy = hull.mean(axis=0)

            # grow polygon; add extra growth for top half (forehead)
            dx, dy = hull[:, 0] - cx, hull[:, 1] - cy
            sy = np.where(hull[:, 1] < cy, scale * top_scale, scale)  # more growth above center
            grown = np.stack([cx + dx * scale, cy + dy * sy], axis=1)
            grown = np.clip(grown, [0, 0], [w - 1, h - 1]).astype(np.int32)

            cv2.fillConvexPoly(mask, grown, 255)

            if dilate_frac > 0:
                k = max(1, int(dilate_frac * max(w, h)))
                k += (k + 1) % 2  # make odd
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.dilate(mask, kernel, iterations=1)

    red = np.full_like(img, (0, 0, 255))
    m3  = cv2.merge([mask, mask, mask])
    out = np.where(m3 > 0, (alpha * red + (1 - alpha) * img).astype(np.uint8), img)
    cv2.imwrite(dst_path, out)

if __name__ == "__main__":
    highlight_face(sys.argv[1], sys.argv[2])

