# create_mask_ref.py
import sys, cv2, numpy as np, mediapipe as mp

from PIL import Image

def _binary_face_mask_from_bgr(
    img: np.ndarray,
    scale: float = 1.05,
    top_scale: float = 1.20,
    dilate_frac: float = 0.04,
    largest_only: bool = True,
) -> np.ndarray:
    """
    Compute a uint8 HxW binary face mask (0/255) from a BGR image.
    Uses mediapipe FaceMesh and the face-oval hull, expanded and dilated.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mesh = mp.solutions.face_mesh

    with mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return mask

    faces = res.multi_face_landmarks

    def hull_for_face(face) -> np.ndarray:
        pts = []
        for i, j in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
            for idx in (i, j):
                lm = face.landmark[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
        if len(pts) < 3:
            return None
        return cv2.convexHull(np.array(pts, np.int32))

    if largest_only and len(faces) > 1:
        best_hull, best_area = None, 0.0
        for f in faces:
            hll = hull_for_face(f)
            if hll is None:
                continue
            area = cv2.contourArea(hll)
            if area > best_area:
                best_hull, best_area = hll, area
        if best_hull is None:
            return mask
        hull = best_hull.reshape(-1, 2).astype(np.float32)
    else:
        first_hull = hull_for_face(faces[0])
        if first_hull is None:
            return mask
        hull = first_hull.reshape(-1, 2).astype(np.float32)

    # grow polygon; add extra growth for top half (forehead)
    cx, cy = hull.mean(axis=0)
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

    return mask


def compute_face_mask_from_pil(
    pil_img: "Image.Image",
    scale: float = 1.05,
    top_scale: float = 1.20,
    dilate_frac: float = 0.04,
    largest_only: bool = True,
) -> np.ndarray:
    """Importable API: returns a uint8 HxW binary mask (0/255) from a PIL image."""
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return _binary_face_mask_from_bgr(bgr, scale=scale, top_scale=top_scale, dilate_frac=dilate_frac, largest_only=largest_only)


def highlight_face(src_path, dst_path, alpha=0.6, scale=1.05, top_scale=1.20, dilate_frac=0.04):
    """
    scale: uniform polygon growth (1.10–1.30)
    top_scale: extra vertical growth for points above center (1.00–1.20)
    dilate_frac: fraction of image size for dilation kernel (0.00–0.06)
    """
    # img = cv2.imread(src_path);  h, w = img.shape[:2]
    # mask = np.zeros((h, w), np.uint8)
    # mesh = mp.solutions.face_mesh

    # with mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as fm:
    #     res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # if res.multi_face_landmarks:
    #     best_hull, best_area = None, 0
    #     for face in res.multi_face_landmarks:
    #         pts = []
    #         for i, j in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
    #             for idx in (i, j):
    #                 lm = face.landmark[idx]
    #                 pts.append([int(lm.x * w), int(lm.y * h)])
    #         hull = cv2.convexHull(np.array(pts, np.int32))
    #         area = cv2.contourArea(hull)
    #         if area > best_area: best_hull, best_area = hull, area

    #     if best_hull is not None:
    #         hull = best_hull.reshape(-1, 2).astype(np.float32)
    #         cx, cy = hull.mean(axis=0)

    #         # grow polygon; add extra growth for top half (forehead)
    #         dx, dy = hull[:, 0] - cx, hull[:, 1] - cy
    #         sy = np.where(hull[:, 1] < cy, scale * top_scale, scale)  # more growth above center
    #         grown = np.stack([cx + dx * scale, cy + dy * sy], axis=1)
    #         grown = np.clip(grown, [0, 0], [w - 1, h - 1]).astype(np.int32)

    #         cv2.fillConvexPoly(mask, grown, 255)

    #         if dilate_frac > 0:
    #             k = max(1, int(dilate_frac * max(w, h)))
    #             k += (k + 1) % 2  # make odd
    #             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    #             mask = cv2.dilate(mask, kernel, iterations=1)

    img = cv2.imread(src_path)
    mask = _binary_face_mask_from_bgr(img, scale=scale, top_scale=top_scale, dilate_frac=dilate_frac, largest_only=True)

    red = np.full_like(img, (0, 0, 255))
    m3  = cv2.merge([mask, mask, mask])
    out = np.where(m3 > 0, (alpha * red + (1 - alpha) * img).astype(np.uint8), img)
    cv2.imwrite(dst_path, out)

if __name__ == "__main__":
    highlight_face(sys.argv[1], sys.argv[2])

