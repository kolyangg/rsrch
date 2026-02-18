from src.utils.id_utils import analyze_faces, create_face_analyzer
import numpy as np


class Aligner():
    def __init__(self):
        self.face_detector = create_face_analyzer(
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
            ctx_id=0,
            det_size=(640, 640),
            fallback_ctx_id=-1,
            quiet=True,
        )

    def __call__(self, input_images):
        bboxes = []
        embeds = []
        for img in input_images:
            max_img_size = max(img.size)
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(self.face_detector, img)
            if len(faces) == 0:
                bboxes.append(None)
                embeds.append(None)
                continue

            face_embeds = []
            face_bboxes = []
            for face in faces:
                bbox = face['bbox'].astype(np.int32)
                bbox = np.clip(bbox, 0, max_img_size)
                face_bboxes.append(bbox.tolist())
                face_embeds.append(face['embedding'])
                
            embeds.append(face_embeds)
            bboxes.append(face_bboxes)
        
        return bboxes, embeds
