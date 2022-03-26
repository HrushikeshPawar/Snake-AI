import torch
from torch.nn import Module, Linear, MSELoss
from torch.nn.functional import relu
from torch.optim import Adam
import numpy as np


class Linear_QNet(Module):

    def __init__(self, input_siz: int, hidden1_size: int, hidden2_size: int, output_size: int) -> None:
        super().__init__()

        if hidden2_size == 0:
            self.Linear1 = Linear(input_siz, hidden1_size)
            self.Linear2 = Linear(hidden1_size, output_size)
        else:
            self.Linear1 = Linear(input_siz, hidden1_size)
            self.Linear2 = Linear(hidden1_size, hidden2_size)
            self.Linear3 = Linear(hidden2_size, output_size)

        self.hidden2_size = hidden2_size

    # Create Forward Neural Network
    def forward(self, x):
        x = relu(self.Linear1(x))

        if self.hidden2_size == 0:
            x = self.Linear2(x)
        else:
            x = relu(self.Linear2(x))
            x = self.Linear3(x)
        return x

    # Save the trained model
    def save(self, fpath : str) -> None:
        """
        Save the model
        """
        torch.save(self.state_dict(), fpath)

    # Load the saved model
    def load(self, fpath : str) -> None:
        """
        Load the model
        """
        self.load_state_dict(torch.load(fpath))

    # Save the checkpoint
    def save_checkPoints(self, state, fpath: str) -> None:

        # Save the checkpoint
        torch.save(state, fpath)

        # If it is the best model so far, save it
        # if is_best:
        #     self.save()

    # Load the checkpoint
    def load_checkPoints(self, fpath: str) -> None:
        """
        Load the model
        """
        # checkpoint = torch.load(fpath)
        # model.load_state_dict(checkpoint['state_dict'])
        # trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        # return model, trainer, checkpoint['epoch']
        pass


# The Main Trainer class
class QTrainer:

    def __init__(self, model, learning_rate, gamma) -> None:

        self.model         = model
        self.gamma         = gamma
        self.learning_rate = learning_rate
        self.optimizer     = Adam(model.parameters(), lr=learning_rate)
        self.criterion     = MSELoss()

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
