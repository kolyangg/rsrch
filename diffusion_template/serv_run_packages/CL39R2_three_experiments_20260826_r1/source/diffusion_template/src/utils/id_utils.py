import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
# pip install insightface==0.7.3
from insightface.app import FaceAnalysis

### 
# https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/165#issue-2055829543
###
class FaceAnalysis2(FaceAnalysis):
    # NOTE: allows setting det_size for each detection call.
    # the model allows it but the wrapping code from insightface
    # doesn't show it, and people end up loading duplicate models
    # for different sizes where there is absolutely no need to
    def get(self, img, max_num=0, det_size=(640, 640)):
        if det_size is not None:
            self.det_model.input_size = det_size

        return super().get(img, max_num)


def create_face_analyzer(
    *,
    providers,
    allowed_modules,
    provider_options=None,
    ctx_id=0,
    det_size=(640, 640),
    fallback_ctx_id=-1,
    quiet=True,
):
    """
    Create and prepare FaceAnalysis2 while optionally suppressing insightface stdout/stderr prints.
    """

    def _build():
        kwargs = {
            "providers": providers,
            "allowed_modules": allowed_modules,
        }
        if provider_options is not None:
            kwargs["provider_options"] = provider_options

        analyzer = FaceAnalysis2(**kwargs)
        try:
            analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception:
            analyzer.prepare(ctx_id=fallback_ctx_id, det_size=det_size)
        return analyzer

    if not quiet:
        return _build()

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return _build()

def analyze_faces(face_analysis: FaceAnalysis, img_data: np.ndarray, det_size=(640, 640)):
    # NOTE: try detect faces, if no faces detected, lower det_size until it does
    detection_sizes = [None] + [(size, size) for size in range(640, 256, -64)] + [(256, 256)]

    for size in detection_sizes:
        faces = face_analysis.get(img_data, det_size=size)
        if len(faces) > 0:
            return faces

    return []
