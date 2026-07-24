from .model import PhotoMakerIDEncoder
# from .model_v2 import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .model_v2_NS import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .resampler import FacePerceiverResampler
# from .pipeline import PhotoMakerStableDiffusionXLPipeline
from .pipeline_NS_ import PhotoMakerStableDiffusionXLPipeline2 # for inference_scripts/attn_hm_NS_nosm7.py
# from .pipeline_NS_old2 import PhotoMakerStableDiffusionXLPipeline2
# from .pipeline_NS_old3 import PhotoMakerStableDiffusionXLPipeline2
# from .pipeline_NS2 import PhotoMakerStableDiffusionXLPipeline2
# from .pipeline_NS2_03Aug import PhotoMakerStableDiffusionXLPipeline2
# from .pipeline_NS2_03Aug2 import PhotoMakerStableDiffusionXLPipeline2
# from .pipeline_NS2_03Aug5 import PhotoMakerStableDiffusionXLPipeline2 as PhotoMakerStableDiffusionXLPipeline
# from .pipeline_NS2_04Aug5 import PhotoMakerStableDiffusionXLPipeline2

# from .pipeline_NS2_04Aug4 import PhotoMakerStableDiffusionXLPipeline2 as PhotoMakerStableDiffusionXLPipeline
# from .pipeline_NS3_v3 import PhotoMakerStableDiffusionXLPipeline

from .pipeline_br import PhotoMakerStableDiffusionXLPipeline


from .pipeline_controlnet import PhotoMakerStableDiffusionXLControlNetPipeline
from .pipeline_t2i_adapter import PhotoMakerStableDiffusionXLAdapterPipeline

# InsightFace Package
from .insightface_package import FaceAnalysis2, analyze_faces

__all__ = [
    "FaceAnalysis2",
    "analyze_faces",
    "FacePerceiverResampler",
    "PhotoMakerIDEncoder",
    "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken",
    "PhotoMakerStableDiffusionXLPipeline",
    "PhotoMakerStableDiffusionXLPipeline2",
    "PhotoMakerStableDiffusionXLControlNetPipeline",
    "PhotoMakerStableDiffusionXLAdapterPipeline",
]