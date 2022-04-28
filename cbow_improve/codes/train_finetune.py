# train.py
# Define the class for training
import time
import numpy as np
import torch
import os
import json

class Trainer():
    '''
    Trainer class
    '''
    def __init__(self, model, epochs, train_dataloader, val_dataloader,
                 criterion, optimizer, lr_scheduler, device, model_dir,
                 model_name):
        '''
        parameters:
        - model: torch.nn.module, model to be trained
        - epochs: number of epochs
        - train_dataloader: DataLoader that contains training data
        - val_dataloader:  DataLoader that contains validation data
        - criterion: loss function, expected to be torch.nn.CrossEntropyLoss()
        - optimizer: optimizer used in training, expected to be classes in torch.optim
        - lr_scheduler: learning rate scheduler, expected to be classes in torch.optim.scheduler
        - device: the training device, expected to be torch.device
        - model_dir: path to save the model
        - model_name: model name, str
        '''

        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.embed_size = self.model.embed_size_1 + self.model.embed_size_2
        self.model.to(self.device)
        self.log_path = os.path.join(self.model_dir, "train_log_{}.json".format(self.embed_size))

    def train(self):
        '''
        Train the model.
        During each epoch:
        - save the training, validation message (including the loss, the learning rate and time) into a json file.
        - save the model if the validation loss is the best.
        '''

        model_info = '# Total parameters: {}'.format(sum(param.numel() for param in self.model.parameters()))
        print(model_info)

        begin_message = "Begin training, embed size: {}, total epochs: {}".format(self.embed_size, self.total_epochs)
        print(begin_message)

        logs = []
        logs.append(begin_message)

        self.model.to(self.device)
        best_loss = 10
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            val_loss = []
            start_time = time.time()
            train_log = {}

            for i, batch_data in enumerate(self.train_dataloader, 1):
                word1 = batch_data[0].to(self.device)
                word2 = batch_data[1].to(self.device)
                labels = batch_data[2].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(word1, word2)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            epoch_train_loss = np.mean(train_loss)
            train_message = '[ Epoch{}, Train ] | Loss:{:.5f} Time:{:.6f}'.format(epoch + 1,
                                                                                  epoch_train_loss,
                                                                                  time.time() - start_time)
            print(train_message)

            self.model.eval()
            start_time = time.time()
            with torch.no_grad():
                for i, batch_data in enumerate(self.val_dataloader , 1):
                    word1 = batch_data[0].to(self.device)
                    word2 = batch_data[1].to(self.device)
                    labels = batch_data[2].to(self.device)

                    outputs = self.model(word1, word2)
                    loss = self.criterion(outputs, labels)

                    val_loss.append(loss.item())

            epoch_val_loss = np.mean(val_loss)
            val_message = '[ Epoch{}, Val ] | Loss:{:.5f} Time:{:.6f}'.format(epoch + 1, epoch_val_loss,
                                                                              time.time() - start_time)
            print(val_message)

            flag = False
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                save_message = 'save model with val loss {:.5f}'.format(epoch_val_loss)
                print(save_message)
                flag = True
                torch.save(self.model, "{}/{}_{}.pt".format(self.model_dir, self.model_name, self.embed_size))

            self.lr_schedule.step()

            train_log["epoch"] = epoch
            train_log["train_message"] = train_message
            train_log["val_message"] = val_message
            train_log["epoch_train_loss"] = epoch_train_loss
            train_log["epoch_val_loss"] = epoch_val_loss
            if flag:
                train_log["save_message"] = save_message
            logs.append(train_log)
            with open(self.log_path, "w") as fp:
                json.dump(logs, fp)
