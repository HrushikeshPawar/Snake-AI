from settings import settings
from model import QTrainer, Linear_QNet
from helper import VISION8, Direction, plot, Point
from game import SnakeGame
import numpy as np
import random
from collections import deque
import torch
from datetime import datetime
import os
import sys


# Game Environment
BLOCKSIZE = settings['block_size']
GRID_H    = settings['grid_height']
GRID_W    = settings['grid_width']
BORDER    = settings['border']
SPEED     = settings['speed']
# FONT      = pygame.font.Font(settings['font'], 20)
VISION    = VISION8


# The model settings
MAX_MEMORY          = settings['max_memory']
BATCH_SIZE          = settings['batch_size']
LEARNING_RATE       = settings['learning_rate']
DISCOUNT_RATE       = settings['discount_rate']
EPSILON_MAX         = settings['epsilon_max']
EPSILON_MIN         = settings['epsilon_min']
EPSILON_DECAY       = settings['epsilon_decay']
EPOCHS              = settings['epochs']
HIDDEN_LAYER1_SIZE  = settings['hidden_layer1_size']
HIDDEN_LAYER2_SIZE  = settings['hidden_layer2_size']

# File Paths
GIF_path          = settings['GIF_path']
Graph_path        = settings['Graph_path']
Model_path        = settings['Model_path']
Checkpoint_path   = settings['Checkpoint_path']
Log_path          = settings['Log_path']


# The Agent class
class Agent:

    def __init__(self) -> None:
        self.n_games            = 0
        self.memory             = deque(maxlen=MAX_MEMORY)
        self.learning_rate      = LEARNING_RATE
        self.discount_rate      = DISCOUNT_RATE
        self.epsilon            = EPSILON_MAX
        self.epsilon_min        = EPSILON_MIN
        self.epsilon_decay      = EPSILON_DECAY
        self.epochs             = EPOCHS
        self.hidden_layer1_size = HIDDEN_LAYER1_SIZE
        self.hidden_layer2_size = HIDDEN_LAYER2_SIZE

        # Initialize the model
        self.model   = Linear_QNet(
            input_siz=3 * len(VISION8) + 4 + 4 + 3,
            # input_siz=3 + 4 + 4 + 4,
            # input_siz=3 + 4 + 4,
            hidden1_size=HIDDEN_LAYER1_SIZE,
            hidden2_size=HIDDEN_LAYER2_SIZE,
            output_size=3
        )  # Neural Network Model
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.discount_rate)

    def get_state(self, game: SnakeGame) -> np.ndarray:

        head = game.snake.head

        # Points near the snake head
        point_l = Point(head.x - BLOCKSIZE, head.y)
        point_r = Point(head.x + BLOCKSIZE, head.y)
        point_u = Point(head.x, head.y - BLOCKSIZE)
        point_d = Point(head.x, head.y + BLOCKSIZE)

        # Bool value of current direction
        dir_l = game.snake.direction == Direction.LEFT
        dir_r = game.snake.direction == Direction.RIGHT
        dir_u = game.snake.direction == Direction.UP
        dir_d = game.snake.direction == Direction.DOWN

        state = []

        # First get all the vision of the snake
        for vis in game.snake.vision:
            state.append(vis.dist_to_wall)
            state.append(vis.is_food_visible)
            state.append(vis.is_self_visible)

        # # Is Dange Straight Ahead?
        state.append((dir_l and game._is_collision(point_l)) or
                     (dir_r and game._is_collision(point_r)) or
                     (dir_u and game._is_collision(point_u)) or
                     (dir_d and game._is_collision(point_d)))

        # Is Danger Right?
        state.append((dir_l and game._is_collision(point_u)) or
                     (dir_r and game._is_collision(point_d)) or
                     (dir_u and game._is_collision(point_r)) or
                     (dir_d and game._is_collision(point_l)))

        # Is Danger Left?
        state.append((dir_l and game._is_collision(point_d)) or
                     (dir_r and game._is_collision(point_u)) or
                     (dir_u and game._is_collision(point_l)) or
                     (dir_d and game._is_collision(point_r)))

        # Then add the direction of snake's head
        state.append(dir_u)
        state.append(dir_r)
        state.append(dir_d)
        state.append(dir_l)

        # # Then add the direction of the tails
        # dir_l = game.snake.tail_direction == Direction.LEFT
        # dir_r = game.snake.tail_direction == Direction.RIGHT
        # dir_u = game.snake.tail_direction == Direction.UP
        # dir_d = game.snake.tail_direction == Direction.DOWN
        # state.append(dir_u)
        # state.append(dir_r)
        # state.append(dir_d)
        # state.append(dir_l)

        # Food Location
        state.append(game.food.x < game.snake.head.x)   # Food is to the left of the snake
        state.append(game.food.y < game.snake.head.y)   # Food is below the snake
        state.append(game.food.x > game.snake.head.x)   # Food is to the right of the snake
        state.append(game.food.y > game.snake.head.y)

        return np.array(state, dtype=float)

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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = 0.0

        action     = [0, 0, 0]

        if random.random() < self.epsilon:
            action[random.randint(0, 2)] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def workers(game, agent, CHK_FILE_PATH, MODEL_FPATH):

    global record, games_played, total_reward, plot_scores, plot_average_scores, total_score, average_score

    while agent.n_games < EPOCHS:

        # Get Old State
        old_state = agent.get_state(game)

        # Get Action
        action = agent.get_action(old_state)

        # Define move in terms of [Straight, Left Turn, Right Turn]
        if game.snake.direction is None:
            move = Direction.RIGHT
        else:
            clock_wise  = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx         = clock_wise.index(game.snake.direction)

            # Check the action and move the snake accordingly
            if np.array_equal(action, np.array([1, 0, 0])):
                move = clock_wise[idx]    # Move straight in the same direction
            elif np.array_equal(action, np.array([0, 1, 0])):
                move = clock_wise[(idx + 1) % 4]    # Turn Right
            else:
                move = clock_wise[(idx - 1) % 4]    # Turn Left

        # Perform Action
        # directions = [game.snake.direction, Direction.RIGHT, Direction.LEFT]
        # move = directions[action.index(1)]
        done, score, reward = game.play_step(move)
        game.n_games        = agent.n_games
        games_played        = agent.n_games
        new_state           = agent.get_state(game)
        total_reward        += reward

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
                'trainer': agent.trainer.optimizer.state_dict(),
                'scores': plot_scores,
                'average_scores': plot_average_scores,
                'total_score': total_score
            }

            # Update Score
            if score > record:
                record = score

                # Save CheckPoint
                agent.model.save_checkPoints(state=checkpoint, fpath=CHK_FILE_PATH)
                agent.model.save(MODEL_FPATH)

            else:
                agent.model.save_checkPoints(state=checkpoint, fpath=CHK_FILE_PATH)

            print(f'Game {agent.n_games} | Score: {score} | Record: {record} | Total Reward: {total_reward} | Current Epsilon {agent.epsilon}')

            # Plotting
            plot_scores.append(score)
            total_score += score
            average_score = total_score / agent.n_games
            plot_average_scores.append(average_score)

            plot_fpath = os.path.join(Graph_path, f'{today} - Game {nth_game} - Grid {GRID_W} X {GRID_H}.png')
            plot(plot_scores, plot_average_scores, plot_fpath, save_plot=True)


def train():

    global record, average_score, total_reward, model, games_played, plot_scores, plot_average_scores, current_epsilon

    # Initialize the agent
    record = 0
    agent  = Agent()
    model = str(agent.model).replace('\n', '\t')
    print(model)

    # Define required variables
    CHK_FILE_PATH = os.path.join(Checkpoint_path, f'{today} - Game {nth_game} - Grid {GRID_W} X {GRID_H}.pth')
    MODEL_FPATH   = os.path.join(Model_path, f'{today} - Game {nth_game} - Grid {GRID_W} X {GRID_H}.pth')
    if os.path.exists(CHK_FILE_PATH):
        checkpoint = torch.load(CHK_FILE_PATH)
        plot_scores         = checkpoint['scores']
        plot_average_scores = checkpoint['average_scores']
        total_score         = checkpoint['total_score']
        agent.model.load_state_dict(checkpoint['state_dict'])
        agent.trainer.optimizer.load_state_dict(checkpoint['trainer'])
        agent.n_games       = checkpoint['epoch']
        agent.epsilon       = checkpoint['epsilon']
    else:
        plot_scores         = []
        plot_average_scores = []
        total_score         = 0

    # Start the game
    game   = SnakeGame(n_game=agent.n_games, is_human=False)

    # Start Training
    while agent.n_games <= EPOCHS:

        # Get Old State
        old_state = agent.get_state(game)

        # Get Action
        action = agent.get_action(old_state)

        # Define move in terms of [Straight, Left Turn, Right Turn]
        if game.snake.direction is None:
            move = Direction.RIGHT
        else:
            clock_wise  = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx         = clock_wise.index(game.snake.direction)

            # Check the action and move the snake accordingly
            if np.array_equal(action, np.array([1, 0, 0])):
                move = clock_wise[idx]    # Move straight in the same direction
            elif np.array_equal(action, np.array([0, 1, 0])):
                move = clock_wise[(idx + 1) % 4]    # Turn Right
            else:
                move = clock_wise[(idx - 1) % 4]    # Turn Left

        # Perform Action
        # directions = [game.snake.direction, Direction.RIGHT, Direction.LEFT]
        # move = directions[action.index(1)]
        done, score, reward = game.play_step(move)
        game.n_games        = agent.n_games
        games_played        = agent.n_games
        new_state           = agent.get_state(game)
        total_reward        += reward
        current_epsilon     = agent.epsilon

        # Training on short memory
        # print(agent.trainer)
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
                'trainer': agent.trainer.optimizer.state_dict(),
                'scores': plot_scores,
                'average_scores': plot_average_scores,
                'total_score': total_score,
                'epsilon': current_epsilon
            }

            # Update Score
            if score > record:
                record = score

                # Save CheckPoint
                agent.model.save_checkPoints(state=checkpoint, fpath=CHK_FILE_PATH)
                agent.model.save(MODEL_FPATH)

            else:
                agent.model.save_checkPoints(state=checkpoint, fpath=CHK_FILE_PATH)

            print(f'Try: {nth_game} | Game {agent.n_games} | Score: {score} | Record: {record} | Total Reward: {total_reward} | Current Epsilon: {agent.epsilon} |')

            # Plotting
            plot_scores.append(score)
            total_score += score
            average_score = total_score / agent.n_games
            plot_average_scores.append(average_score)

            plot_fpath = os.path.join(Graph_path, f'{today} - Game {nth_game} - Grid {GRID_W} X {GRID_H}.png')
            plot(plot_scores, plot_average_scores, plot_fpath, save_plot=True)

    # return record, plot_average_scores[-1]


if __name__ == "__main__":

    # Initialize the logging of results
    today = datetime.today().strftime('%Y%m%d')
    log_fpath = os.path.join(Log_path, f'{today}.log')
    nth_game = 26

    # Initialize the logging of results
    record          = 0
    average_score   = 0
    total_reward    = 0
    games_played    = 0
    model           = ''
    current_epsilon = 0

    try:
        train()
    except KeyboardInterrupt:
        print('\n\nTraining Interrupted')
    except Exception as e:
        print(e)
        print('\n\nTraining Interrupted')
        sys.exit(1)
    finally:
        print('Logging Data...')
        with open(log_fpath, 'r') as f:
            data = f.read()
            data += f'Game {nth_game} | Grid {GRID_W} X {GRID_H} | Record: {record} | Average Score: {average_score} | Games Played: {games_played} | Learning Rate: {LEARNING_RATE} | Discount Rate: {DISCOUNT_RATE} | Epsilon Decay: {EPSILON_DECAY} | Current Epsilon: {current_epsilon} | Batch Size: {BATCH_SIZE} | Hidden Layer 1: {HIDDEN_LAYER1_SIZE} | Hidden Layer 2: {HIDDEN_LAYER2_SIZE} | NN Model: {model} |\n'
        with open(log_fpath, 'w') as f:
            f.write(data)
        print('\n\nExiting...')
        sys.exit(0)
