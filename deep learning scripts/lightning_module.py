import pandas as pd
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


class UrbanSound8KModule(pl.LightningModule):
    def __init__(self, n_classes, classes_map, optimizer, optimizer_parameters, lr_scheduler, lr_scheduler_parameters, learning_rate, batch_size, model):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=["model"])   
        self.model = model     

        # Instantiation of the metrics
        self.train_accuracy = Accuracy(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_accuracy = Accuracy(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_precision = Precision(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_recall = Recall(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_f1_score = F1Score(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_confmat = ConfusionMatrix(num_classes=self.hparams.n_classes, normalize="true")           
        

    def configure_optimizers(self): 
        if self.hparams.optimizer == "Adam": 
            optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.lr_scheduler == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.lr_scheduler_parameters["patience"], verbose=True)  

        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "validation_loss",
                        "frequency": 1
                        }
                }
    
        
    def training_step(self, train_batch, batch_idx): 
        index, audio_name, targets, inputs = train_batch
        logits = self.model(inputs) 
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        self.train_accuracy(logits, targets)
        self.log("training_loss", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("training_accuracy", self.train_accuracy, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)        
        return {"inputs":inputs, "targets":targets, "predictions":predictions, "loss":loss}
    
    
    def training_epoch_end(self, outputs):
        # Log weights and biases for all layers of the model
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,self.current_epoch)
        # Only after the first training epoch, log one of the training inputs as a figure and log the model graph
        if self.current_epoch == 0:
            input_sample = outputs[0]["inputs"][0]
            input_sample_target = outputs[0]["targets"][0].item()
            input_sample_class = self.hparams.classes_map[input_sample_target]
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111)
            ax.imshow(torch.squeeze(input_sample).cpu(), cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(f"Class: {input_sample_class}")
            ax.set_xlabel("Time Frames")
            self.logger.experiment.add_figure(f"Training sample input", fig)
            input_sample = torch.unsqueeze(input_sample, 3)
            input_sample = torch.permute(input_sample, (3,0,1,2))
            self.logger.experiment.add_graph(self.model, input_sample)

            
    def validation_step(self, validation_batch, batch_idx):
        index, audio_name, targets, inputs = validation_batch
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        self.validation_accuracy(predictions, targets)
        self.validation_precision(predictions, targets)
        self.validation_recall(predictions, targets)
        self.validation_f1_score(predictions, targets)
        self.validation_confmat.update(predictions, targets)
        self.log("hp_metric", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("validation_accuracy", self.validation_accuracy, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("validation_precision", self.validation_accuracy, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("validation_recall", self.validation_accuracy, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("validation_f1_score", self.validation_accuracy, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        return {"inputs":inputs, "targets":targets, "predictions":predictions, "loss":loss}
    
    
    def validation_epoch_end(self, outputs):
        # Compute the confusion matrix, turn it into a DataFrame, generate the plot and log it
        cm = self.validation_confmat.compute()
        cm = cm.cpu()
        self.validation_confmat.reset()
        df_cm = pd.DataFrame(cm.numpy(), index=range(self.hparams.n_classes), columns=range(self.hparams.n_classes))
        fig, ax = plt.subplots(figsize=(16,16))
        sns.heatmap(data=df_cm, annot=True, cmap="Blues", ax=ax, vmin=0, vmax=1)
        ax.set_xticklabels(labels=list(self.hparams.classes_map.values()))
        ax.set_yticklabels(labels=list(self.hparams.classes_map.values()))
        ax.tick_params(axis="y", labelrotation=0)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Real", fontsize=12)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        
    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()