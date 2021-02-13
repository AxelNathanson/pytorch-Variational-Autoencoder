import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import BetaVAE

torch.set_default_dtype(torch.float)


class Solver(object):
    def __init__(self, model: BetaVAE, train_set, validation_set, **kwargs):
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set

        # Optional arguments
        learning_rate = kwargs.pop('learning_rate', 1e-4)
        weight_decay = kwargs.pop('weight_decay', 3e-4)
        self.print_every = kwargs.pop('print_every', 200)
        self.batch_size = kwargs.pop('batch_size', 64)
        lr_rate_decay = kwargs.pop('lr_rate_decay', 0.5)
        lr_decay_epochs = kwargs.pop('decay_every_', 1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=lr_decay_epochs, 
                                                         gamma=lr_rate_decay)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Reset all of my saved history
        self._reset()
        self.epoch_acc = None

    def _reset(self):
        """Resets all the history variables of the training.
        """        
        self.train_loss_history = []
        self.val_loss_history = []
    
    def _step(self, image):
        """Performs one step of the function, given an image and target
        
        Arguments:
            image {tensor} -- Input to model
            target {array} -- Target number
        """     
        # Forward pass
        output, (mu, logsigma) = self.model(image)

        loss = self.model.compute_loss(image, output, mu, logsigma)
        self.train_loss_history.append(loss.detach.numpy())

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_epochs=10):
        print(f'Started training. Will run for: {num_epochs} Epochs.',
              f'Iterations per Epoch: {int(len(self.train_set)/self.batch_size) + 1}.')

        train_loader = DataLoader(self.train_set,
                                  batch_size=self.batch_size, 
                                  shuffle=True)

        validation_loader = DataLoader(self.validation_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)

        for epoch in range(num_epochs):
            self.epoch_acc = []
            self.model.train()
            for it, data in tqdm(enumerate(train_loader)):
                if it % self.print_every == 0 and it != 0:
                    print(f'Done with iteration: {it}/{len(train_loader)}.')
                
                image, _ = data
                image.to(self.device)
             
                self._step(image)

            self.model.eval()

            print('Computing Validation Loss')
            for batch, data in tqdm(enumerate(validation_loader)):
                image, _ = data
                image.to(self.device)

                output, (mu, sigma) = self.model(image)
                loss = self.model.compute_loss(image, output, mu, sigma)
                self.val_loss_history.append(loss.detach().numpy())

            # Finnish the epoch with updating the lr_rate.
            self.scheduler.step()
