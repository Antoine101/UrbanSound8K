import pandas as pd
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


class UrbanSound8KModule(pl.LightningModule):
    def __init__(self, n_classes, classes_map, learning_rate, batch_size, model):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters(ignore=["model"])   
        self.model = model     

        # Instantiation of the metrics
        self.train_accuracy = Accuracy(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_accuracy = Accuracy(num_classes=self.hparams.n_classes, average="weighted")
        self.validation_confmat = ConfusionMatrix(num_classes=self.hparams.n_classes)           
        

    def configure_optimizers(self):  
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)   
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
        self.log("training_loss", loss, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("training_accuracy", self.train_accuracy, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)        
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
            ax.set_ylabel("Mel Bands")
            self.logger.experiment.add_figure(f"Training sample input", fig)
            input_sample = torch.unsqueeze(input_sample, 3)
            input_sample = torch.permute(input_sample, (3,0,1,2))
            self.logger.log_graph(self, input_sample)

            
    def validation_step(self, validation_batch, batch_idx):
        # Unpack the validation batch
        index, audio_name, targets, inputs = validation_batch
        # Pass the inputs to the model to get the logits
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        self.validation_accuracy(predictions, targets)
        self.validation_confmat.update(predictions, targets)
        # Log the loss and the accuracy
        self.log("validation_loss", loss, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log("validation_accuracy", self.validation_accuracy, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
        return {"inputs":inputs, "targets":targets, "predictions":predictions, "loss":loss, "audio_name":audio_name}
    
    
    def validation_epoch_end(self, outputs):
        # Concatenate the predictions of all batches
        preds = torch.cat([output["predictions"] for output in outputs])
        # Concatenate the targets of all batches
        targets = torch.cat([output["targets"] for output in outputs])
        # Concatenate the audios name of all batches
        audio_name_tuples_list = [output["audio_name"] for output in outputs]
        audio_name = [audio_name for audio_name_tuples in audio_name_tuples_list for audio_name in audio_name_tuples]
        for i in range(len(preds)):
            self.logger.experiment.add_text("Predictions on validation set", f"Audio: {audio_name[i]} - Class: {targets[i]} - Predicted: {preds[i]}")
        # Compute the confusion matrix, turn it into a DataFrame, generate the plot and log it
        cm = self.validation_confmat.compute()
        cm = cm.cpu()
        self.validation_confmat.reset()
        for class_id in range(self.hparams.n_classes):
                precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])
                precision = round(precision.item()*100,1)
                self.log(f"validation_precision/{class_id}", precision)
                recall = cm[class_id, class_id] / torch.sum(cm[class_id,:])
                recall = round(recall.item()*100,1)
                self.log(f"validation_recall/{class_id}", recall)
        df_cm = pd.DataFrame(cm.numpy(), index = range(self.hparams.n_classes), columns=range(self.hparams.n_classes))
        plt.figure()
        fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.yticks(rotation=0)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        
    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()