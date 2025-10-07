import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class SDXLTrainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, train_metrics: MetricTracker):
        """
        Run batch through the model, compute loss,
        and do training step.

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            train_metrics (MetricTracker): MetricTracker object that computes
                and aggregates training losses.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        self.optimizer.zero_grad()
            
        do_cfg =  (batch["batch_idx"] % self.cfg_step == 0)
        output = self.model(**batch, do_cfg=do_cfg)
        batch.update(output)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        if self.is_train:
            assert torch.isfinite(batch["loss"]) # sum of all losses is always called loss
            self.accelerator.backward(batch["loss"]) 
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
            train_metrics.update(loss_name, batch[loss_name].item())

        return batch

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        seed = batch.get("seed", self.config.validation_args.get("seed", 0))
        generator = torch.Generator(device='cpu').manual_seed(seed)
        generated_images = self.pipe(
            prompt=batch['prompt'],
            generator=generator,
            **self.config.validation_args
        ).images

        batch['generated'] = generated_images

        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)
                
        return batch
        
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            prompt = batch['prompt']
            generated_img = batch['generated'][0].resize((256, 256))

            cutted_prompt = prompt.replace(" ", "_")[:30]
            image_name = f"{mode}_images/{cutted_prompt}"
            self.writer.add_image(image_name, generated_img)


class PhotomakerLoraTrainer(SDXLTrainer):
    def __init__(self, masked_loss_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_loss_step = masked_loss_step
        
    def process_batch(self, batch, train_metrics: MetricTracker):
        self.optimizer.zero_grad()
            
        do_cfg = (batch["batch_idx"] % self.cfg_step == 0)
        output = self.model(**batch, do_cfg=do_cfg)
        batch.update(output)

        batch["is_masked_loss"] = (batch["batch_idx"] % self.masked_loss_step == 0)
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        if self.is_train:
            assert torch.isfinite(batch["loss"]) # sum of all losses is always called loss
            self.accelerator.backward(batch["loss"]) 
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            batch[loss_name] = self.accelerator.gather(batch[loss_name]).mean()
            train_metrics.update(loss_name, batch[loss_name].item())

        return batch
        
    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        seed = self.config.validation_args.get("seed", 0)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        generated_images = self.pipe(
            prompt=batch['prompt'],
            generator=generator,
            input_id_images=batch['ref_images'],
            **self.config.validation_args
        ).images

        batch['generated'] = generated_images

        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)
                
        return batch
