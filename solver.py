import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import BetaVAE

torch.set_default_dtype(torch.float)


class Solver(object):
    def __init__(self, model: BetaVAE, train_set, validation_set=None, test_set=None, **kwargs):
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set

        # Optional arguments
        learning_rate = kwargs.pop('learning_rate', 1e-4)
        weight_decay = kwargs.pop('weight_decay', 0)
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

        self.train_kl_loss = []
        self.train_mse_loss = []
        self.val_kl_loss = []
        self.val_mse_loss = []

    def _train_step(self, image):
        """Performs one training step given an image.
        
        Arguments:
            image {tensor} -- Input to model
        """     
        # Forward pass
        output, (mu, logsigma) = self.model(image)

        loss, (mse_loss, kl_div) = self.model.compute_loss(image, output, mu, logsigma)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss_history.append(loss.item())
        self.train_kl_loss.append(kl_div.item())
        self.train_mse_loss.append(mse_loss.item())

    def train(self, num_epochs=10):
        train_loader = DataLoader(self.train_set,
                                  batch_size=self.batch_size, 
                                  shuffle=True)

        validation_loader = DataLoader(self.validation_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=True)

        for epoch in range(num_epochs):
            with tqdm(train_loader, desc='Training', position=0, leave=True) as Batches:
                self.model.train()
                for data in Batches:
                    image, _ = data
                    image.to(self.device)

                    self._train_step(image)
                    Batches.set_postfix(Epoch=f'{epoch+1}/{num_epochs}',
                                        Loss=torch.tensor(self.train_loss_history[-10:]).mean().item(),
                                        KL_loss=torch.tensor(self.train_kl_loss[-10:]).mean().item(),
                                        MSE_loss=torch.tensor(self.train_mse_loss[-10:]).mean().item())

                self.model.eval()
                with tqdm(validation_loader, desc='Validation', position=0, leave=True) as validation:
                    for data in validation:
                        image, _ = data
                        image.to(self.device)

                        output, (mu, logsigma) = self.model(image)

                        loss, (mse_loss, kl_div) = self.model.compute_loss(image, output, mu, logsigma)

                        self.val_loss_history.append(loss.item())
                        self.val_kl_loss.append(kl_div.item())
                        self.val_mse_loss.append(mse_loss.item())

                        validation.set_postfix(Epoch=f'{epoch+1}/{num_epochs}',
                                               Loss=torch.tensor(self.val_loss_history[-10:]).mean().item(),
                                               KL_loss=torch.tensor(self.val_kl_loss[-10:]).mean().item(),
                                               MSE_loss=torch.tensor(self.val_mse_loss[-10:]).mean().item())

                # Finnish the epoch with updating the lr_rate.
                self.scheduler.step()

    def overtrain_sample(self, num_epochs=10):
        train_loader = DataLoader(self.train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        for epoch in range(num_epochs):
            with tqdm(train_loader, desc='Training', position=0, leave=True) as Batches:
                self.model.train()
                for data in Batches:
                    image, _ = data
                    image.to(self.device)

                    self._train_step(image)
                    Batches.set_postfix(Epoch=f'{epoch + 1}/{num_epochs}',
                                        Loss=torch.tensor(self.train_loss_history[-10:]).mean().item(),
                                        KL_loss=torch.tensor(self.train_kl_loss[-10:]).mean().item(),
                                        MSE_loss=torch.tensor(self.train_mse_loss[-10:]).mean().item())

                # Finnish the epoch with updating the lr_rate.
                self.scheduler.step()

    def predict(self):
        test_loader = DataLoader(self.test_set,
                                 batch_size=len(self.test_set),
                                 shuffle=True)

        for image, target in test_loader:
            z = self.model.sample_latent_space(image)

        return z

    def plot_training_loss(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        ax[0].plot(self.train_loss_history, label='Total loss')
        ax[1].plot(self.train_kl_loss, label='KL loss')
        ax[0].plot(self.train_mse_loss, label='MSE loss')
        for axis in ax:
            axis.legend()