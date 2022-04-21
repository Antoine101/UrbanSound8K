import pandas as pd
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


class Pipeline(pl.LightningModule):
    def __init__(self, out_dim, classes_map, learning_rate, batch_size):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()        
        # Definition of the model
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3)
            )
        self.flatten = nn.Sequential(nn.Flatten()) 
        self.dense = nn.Sequential(
            nn.Linear(128 * 16 * 86, 512),
            nn.Linear(512, 256),
            nn.Linear(256, out_dim)
            )
        # Instantiation of the metrics
        self.accuracy = Accuracy(num_classes=len(classes_map), average="weighted")
        self.recall = Recall(num_classes=len(classes_map), average="weighted")
        self.f1_score = F1(num_classes=len(classes_map), average="weighted")
        self.confmat = ConfusionMatrix(num_classes=len(classes_map))           
        # Instantiation of the classes map
        self.classes_map = classes_map
        # Instantiation of the number of classes
        self.n_classes = len(classes_map)
        # Instatiation of the learning rate
        self.learning_rate = learning_rate
        # Instantiation of the batch size (needed to avoid batch size inference error caused by text returned by the dataset)
        self.batch_size = batch_size
        
    def configure_optimizers(self):  
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)   
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "validation_loss",
                        "frequency": 1
                        }
                }
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits 
    
        
    def training_step(self, train_batch, batch_idx): 
        # Unpack the training batch
        _, _, targets, inputs = train_batch
        # Pass the inputs to the model to get the logits
        logits = self(inputs)
        # Compute the loss
        loss = F.cross_entropy(logits, targets)
        # Get the probabilities for each class by applying softmax
        probs = F.softmax(logits, dim=1)
        # Get the prediction for each batch sample
        _, preds = torch.max(probs, 1)
        # Compute the accuracy
        accuracy = self.accuracy(logits, targets)
        # Log the loss
        self.log("training_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return {"inputs":inputs, "targets":targets, "predictions":preds, "loss":loss}
    
    
    def training_epoch_end(self, outputs):
        # Log weights and biases for all layers of the model
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,self.current_epoch)
        # Only after the first training epoch, log one of the training inputs as a figure and log the model graph
        if self.current_epoch == 0:
            input_sample = outputs[0]["inputs"][0]
            input_sample_target = outputs[0]["targets"][0].item()
            input_sample_class = self.classes_map[input_sample_target]
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
        _, audios_name, targets, inputs = validation_batch
        # Pass the inputs to the model to get the logits
        logits = self(inputs)
        # Compute the loss and log it for early stopping monitoring
        loss = F.cross_entropy(logits, targets)
        # Get the probabilities for each class by applying softmax
        probs = F.softmax(logits, dim=1)
        # Get the prediction for each batch sample
        _, preds = torch.max(probs, 1)
        # Compute the accuracy for this batch
        accuracy = self.accuracy(preds, targets)
        # Log the loss and the accuracy
        self.log("validation_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("validation_accuracy", accuracy, on_step=True, on_epoch=True, batch_size=self.batch_size)
        return {"inputs":inputs, "targets":targets, "predictions":preds, "loss":loss, "audios_name":audios_name}
    
    
    def validation_epoch_end(self, outputs):
        # Concatenate the predictions of all batches
        preds = torch.cat([output["predictions"] for output in outputs])
        # Concatenate the targets of all batches
        targets = torch.cat([output["targets"] for output in outputs])
        # Concatenate the audios name of all batches
        audios_name_tuples_list = [output["audios_name"] for output in outputs]
        audios_name = [audio_name for audios_name_tuples in audios_name_tuples_list for audio_name in audios_name_tuples]
        for i in range(len(preds)):
            self.logger.experiment.add_text("Predictions on validation set", f"Audio: {audios_name[i]} - Class: {targets[i]} - Predicted: {preds[i]}")
        # Compute the confusion matrix, turn it into a DataFrame, generate the plot and log it
        cm = self.confmat(preds, targets)
        cm = cm.cpu()
        for class_id in range(self.n_classes):
                precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])
                precision = round(precision.item()*100,1)
                self.log(f"validation_precision/{class_id}", precision)
                recall = cm[class_id, class_id] / torch.sum(cm[class_id,:])
                recall = round(recall.item()*100,1)
                self.log(f"validation_recall/{class_id}", recall)
        df_cm = pd.DataFrame(cm.numpy(), index = range(self.n_classes), columns=range(self.n_classes))
        plt.figure()
        fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.yticks(rotation=0)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        
    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()