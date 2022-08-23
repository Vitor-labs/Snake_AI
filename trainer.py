import random
import torch
import numpy
from collections import deque

from game import SnakeGame, Direction, Point
from model import Line_Q_net, DeepQTrainer
from ploter import plot

MEMORY_LIMIT = 100_999
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self,):
        self.game_number = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MEMORY_LIMIT)
        self.model = Line_Q_net(11, 256, 3)
        self.trainer = DeepQTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        left = Point(head.x - 20, head.y)
        right = Point(head.x + 20, head.y)
        up = Point(head.x, head.y - 20)
        down = Point(head.x, head.y + 20)
        
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_right and game.is_collision(right)) or 
            (dir_left and game.is_collision(left)) or 
            (dir_up and game.is_collision(up)) or 
            (dir_down and game.is_collision(down)),

            # Danger right
            (dir_up and game.is_collision(right)) or 
            (dir_down and game.is_collision(left)) or 
            (dir_left and game.is_collision(up)) or 
            (dir_right and game.is_collision(down)),

            # Danger left
            (dir_down and game.is_collision(right)) or 
            (dir_up and game.is_collision(left)) or 
            (dir_right and game.is_collision(up)) or 
            (dir_left and game.is_collision(down)),
            
            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return numpy.array(state, dtype=int)

    def remember(self, state, next_state, action, reward, done):
        self.memory.append(
            (state, action, reward, next_state, done)
            ) # popleft if MEMORY_LIMIT is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, next_state, action, reward, done):
        self.trainer.train(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.game_number
        
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    scores = []
    plot_scores = []
    total = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        old_state = agent.get_state(game)
        final_move = agent.get_action(old_state)
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, new_state, final_move, reward, done)
        agent.remember(old_state, new_state, final_move, reward, done)

        if done: # experience replay
            game.reset_game()
            agent.game_number += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print( 'Game: ', agent.game_number, '\n',
                  'Score: ', score, '\n',
                  'Record: ', record, '\n')

            scores.append(score)
            total += score
            mean_score = total / agent.game_number
            plot_scores.append(mean_score)
            
            plot(scores, plot_scores)
            

if __name__ == '__main__':
    train()
