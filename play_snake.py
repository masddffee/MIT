import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
import os

class Agent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)
        self.load_model()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def load_model(self, file_name='model.pth'):
        file_name = os.path.join('./model', file_name)
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name))
            self.model.eval()
            print(f"Loaded model from {file_name}")
        else:
            print(f"Error: No saved model found at {file_name}. Please run train_snake.py first.")
            exit(1)

def play():
    agent = Agent()
    game = SnakeGameAI(False)  # 展示模式，有視窗
    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)
        if done:
            print(f"Game Over! Score: {score}")
            game.reset()

if __name__ == '__main__':
    play()