import torch
from torch import nn
from torch.nn.modules import Module
import torch.optim as optim
import torch.nn.functional as Funtional
import os
import shutil
import numpy as np


class Linear_QNet(Module):

    def __init__(self, input_siz: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.Linear1 = nn.Linear(input_siz, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = Funtional.relu(self.Linear1(x))
        x = self.Linear2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        """
        Save the model
        """
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        torch.save(self.state_dict(), os.path.join(model_folder, file_name))

    def save_checkPoints(self, state, checkpoint_dir, checkpoint_name, best_model_fpath, is_best=False) -> None:

        # Save the model
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        fpath = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(state, fpath)

        # If it is the best model so far, save it
        # if not os.path.exists(best_model_dir):
        #     os.mkdir(best_model_dir)
        if is_best:
            # best_fpath = os.path.join(best_model_dir, f'best_model - ({suffix}).pth')
            shutil.copyfile(fpath, best_model_fpath)

    def load_checkPoints(self, model, trainer, checkpoint_fpath) -> None:
        """
        Load the model
        """
        if not os.path.exists(checkpoint_fpath):
            raise FileNotFoundError('No model to load')

        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        return model, trainer, checkpoint['epoch']


class QTrainer:

    def __init__(self, model, learning_rate, gamma) -> None:

        self.lr         = learning_rate
        self.gamma      = gamma
        self.model      = model
        self.optimizer  = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion  = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> None:

        # Inputs can be single or batch
        # Hence convert them into tensor
        state       = torch.tensor(np.array(state), dtype=torch.float)
        action      = torch.tensor(action, dtype=torch.long)
        reward      = torch.tensor(reward, dtype=torch.float)
        next_state  = torch.tensor(np.array(next_state), dtype=torch.float)

        # If its just single input
        if len(state.shape) == 1:

            # Input will be of the form (1, x)
            state       = torch.unsqueeze(state, dim=0)
            action      = torch.unsqueeze(action, dim=0)
            reward      = torch.unsqueeze(reward, dim=0)
            next_state  = torch.unsqueeze(next_state, dim=0)
            done        = (done, )

        # 1. Predicted Q values with current state
        pred = self.model(state)

        # 2. Predicted Q values with next state - r + gamma * max(pred)
        target = pred.clone()
        for idx in range(len(done)):

            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action).item()] = Q_new

        # 3. Calculate loss
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        # 4. Update the weights
        self.optimizer.step()
