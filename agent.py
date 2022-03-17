import torch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGame, Direction, Point
from model import QTrainer, Linear_QNet
from helper import plot


# Defining Constants
MAX_MEMORY      = 100_000
BATCH_SIZE      = 1000
LEARNING_RATE   = 0.001
BLOCKSIZE       = 20
CHK_FILE_PATH   = 'checkpoints/checkpoint.pth'
CHK_DIR         = 'checkpoints'
BEST_MODEL_DIR  = 'best_model'


class Agent:

    def __init__(self) -> None:
        self.n_games = 0    # Number of Games
        self.epsilon = 0  # Exploration Rate (Controls how much randomness is added to the action)
        self.gamma   = 0.99  # Discount Rate (Controls how much importance is given to future rewards)
        self.memory  = deque(maxlen=MAX_MEMORY)  # Memory (Stores the past experiences)
        self.model   = Linear_QNet(input_siz=11, hidden_size=256, output_size=3)  # Neural Network Model
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)  # Optimizer

        # Load Saved checkpoint
        try:
            self.model, self.trainer, self.n_games = self.model.load_checkPoints(self.model, self.trainer, CHK_FILE_PATH)
        except FileNotFoundError:
            pass

    def get_state(self, game: SnakeGame) -> np.ndarray:

        head = game.head

        # Points near the snake head
        point_l = Point(head.x - BLOCKSIZE, head.y)
        point_r = Point(head.x + BLOCKSIZE, head.y)
        point_u = Point(head.x, head.y - BLOCKSIZE)
        point_d = Point(head.x, head.y + BLOCKSIZE)

        # Bool value of current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State of the game
        state = np.array([

            # Is Dange Straight Ahead?
            (dir_l and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Is Danger Right?
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)) or
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)),

            # Is Danger Left?
            (dir_l and game._is_collision(point_d)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_d and game._is_collision(point_r)),

            # Current Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x,   # Food is to the left of the snake
            game.food.y < game.head.y,   # Food is below the snake
            game.food.x > game.head.x,   # Food is to the right of the snake
            game.food.y > game.head.y,   # Food is above the snake
        ], dtype=int)

        return state

    def remember(self, state: np.ndarray, action, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:

        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: list, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> list:

        # Random Moves = Tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train():

    # Define required variables
    try:
        checkpoint = torch.load(CHK_FILE_PATH)
        plot_scores         = checkpoint['scores']
        plot_average_scores = checkpoint['average_scores']
        total_score         = checkpoint['total_score']
    except FileNotFoundError:
        plot_scores         = []
        plot_average_scores = []
        total_score         = 0

    record              = 0
    agent               = Agent()
    game                = SnakeGame(n_games=agent.n_games)

    # Start Training
    while True:

        # Get Old State
        old_state = agent.get_state(game)

        # Get Action
        action = agent.get_action(old_state)

        # Perform Action
        reward, done, score = game.play_step(action)
        game.n_games = agent.n_games
        new_state           = agent.get_state(game)

        # Training on short memory
        agent.train_short_memory(old_state, action, reward, new_state, done)

        # Remember
        agent.remember(old_state, action, reward, new_state, done)

        # Train on long memory
        if done:

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Create CheckPoint
            checkpoint = {
                'epoch': agent.n_games,
                'state_dict': agent.model.state_dict(),
                'optimizer': agent.trainer.optimizer.state_dict(),
                'scores': plot_scores,
                'average_scores': plot_average_scores,
                'total_score': total_score
            }

            # Update Score
            if score > record:
                record = score

                # Save CheckPoint
                agent.model.save_checkPoints(checkpoint, CHK_DIR, BEST_MODEL_DIR, is_best=True)

            else:
                agent.model.save_checkPoints(checkpoint, CHK_DIR, BEST_MODEL_DIR)

            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')

            # Plotting
            plot_scores.append(score)
            total_score += score
            average_score = total_score / agent.n_games
            plot_average_scores.append(average_score)
            plot(plot_scores, plot_average_scores)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print('\n\nExiting...')
