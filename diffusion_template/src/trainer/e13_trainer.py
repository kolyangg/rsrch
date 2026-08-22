"""Selected E13 trainer settings over the unchanged June trainer API."""

from .sdxl_trainers import PhotomakerLoraTrainer


class E13PhotomakerLoraTrainer(PhotomakerLoraTrainer):
    def __init__(
        self,
        *args,
        post_backward_parameter_touch=True,
        grad_norm_log_only=False,
        face_quality=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.post_backward_parameter_touch = bool(post_backward_parameter_touch)
        self.grad_norm_log_only = bool(grad_norm_log_only)
        self.face_quality_config = face_quality
