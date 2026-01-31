import torch
from src.metrics.base_metric import BaseMetric
import clip

class TextSimMetric(BaseMetric):
    def __init__(self, model_name, device, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__(*args, **kwargs)
        # NOTE: metrics are instantiated on *all* ranks (see train.py),
        # but evaluation runs only on the main process. Loading CLIP on CUDA
        # here can OOM before training even starts. Always load on CPU first,
        # then optionally move to CUDA in `to_cuda()`.
        self.device = device  # desired CUDA device (torch.device) for optional acceleration
        self._active_device = torch.device("cpu")
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.model.eval()

    def to_cpu(self):
        self.model = self.model.to("cpu")
        self._active_device = torch.device("cpu")

    def to_cuda(self):
        # Best-effort move to CUDA; if it OOMs, keep running on CPU.
        try:
            self.model = self.model.to(self.device)
            self._active_device = torch.device(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[TextSimMetric] CUDA OOM while moving CLIP to {self.device}; using CPU for this metric.")
                self.model = self.model.to("cpu")
                self._active_device = torch.device("cpu")
            else:
                raise

    def __call__(self, **batch):
        prompt = batch['prompt']
        tokenized_prompt = clip.tokenize([prompt], truncate=True).to(self._active_device)
        generated = batch['generated']
        assert type(generated) is list, type(generated) 
        assert type(prompt) is str, type(prompt)

            
        preprecessed = [self.preprocess(img) for img in generated]
        images = torch.stack(preprecessed).to(self._active_device) 
        _, logits_per_text = self.model(images, tokenized_prompt)
        result = {"text_sim": logits_per_text.mean().item()}
        return result
