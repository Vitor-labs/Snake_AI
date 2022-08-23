import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as func
import os

class Line_Q_net(nn.Module):
    def __init__(self, input, hidden, output=None):
        super().__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self,x):
        x = func.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save(self, filename='model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        filename= os.path.join(model_path, filename)
        torch.save(self.state_dict(), filename)

class DeepQTrainer:
    def __init__(self,model,lr,gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        predict = self.model(state)
        target = predict.clone()

        for i in range(len(done)):
            new_q = reward[i]
            if not done[i]:
                new_q = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
        
            target[i][torch.argmax(action).item()] = new_q

        self.optimizer.zero_grad()
        loss = self.criterion(target,predict)
        loss.backward
        self.optimizer.step()