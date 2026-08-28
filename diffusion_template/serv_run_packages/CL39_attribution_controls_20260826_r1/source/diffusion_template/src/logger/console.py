from datetime import datetime

import pandas as pd


class ConsoleWriter:
    """Minimal no-network writer for sealed validation-only jobs."""

    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "general/steps_per_sec",
                (self.step - previous_step) / duration.total_seconds(),
            )
            self.timer = datetime.now()

    def add_scalar(self, scalar_name, scalar):
        self.logger.info(f"Step {self.step}: {scalar_name} = {scalar}")

    def add_scalars(self, scalars):
        for scalar_name, scalar_value in scalars.items():
            self.logger.info(f"Step {self.step}: {scalar_name} = {scalar_value}")

    def add_image(self, image_name, image):
        pass

    def add_audio(self, audio_name, audio, sample_rate=None):
        pass

    def add_text(self, text_name, text):
        pass

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        pass

    def add_table(self, table_name, table: pd.DataFrame):
        pass

    def add_images(self, images_name, images):
        pass

