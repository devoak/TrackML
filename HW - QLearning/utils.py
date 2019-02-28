from config import *
import numpy as np


class OneStateHolder:
    def __init__(self):
        self.screens = []
    
    def pushScreen(self, screen):
        if len(self.screens) < 4:
            self.screens.append(stateFromScreen(screen))
        else:
            self.screens.pop()
            self.screens.append(stateFromScreen(screen))

    def getState(self):
        _, H, W = self.screens[0].size()
        state = torch.zeros(4, H, W)
        for j in range(4):
            state[j, :, :] = self.screens[j]
        return state.to(DEVICE)

    def clear(self):
        self.screens = []


    def initWithFirstScreens(self, env = ENVIRONMENT):
        """
            initialise state holder with 4 first screens from
            environment given by the influence just doing nothing actions
        """
        self.clear()
        action = 0
        for i in range(4):
            screen, _, _, _ = env.step(action)
            self.pushScreen(screen)


"""
   Game Memory class for holding all transitions have been made
   
   TODO - upgrade performance
   
"""
class GameMemory:
    
    def __init__(self, replayMemoryCapacity = REPLAY_MEMORY):
        self.totalMemoryCapacity = replayMemoryCapacity
        self.screens = []
        self.actions = []
        self.rewards = []
        self.isTerminal = []
        self.actualMemoryLen = 0
        self.actualInputPosition = 0
    
    def pushScreenActionReward(self, screen, action, reward, isTerminal):
        if self.actualMemoryLen < self.totalMemoryCapacity:
            self.screens.append(stateFromScreen(screen))
            self.actions.append(action)
            self.rewards.append(reward)
            self.isTerminal.append(isTerminal)
            self.actualMemoryLen += 1
        else:
            self.screens[self.actualInputPosition] = stateFromScreen(screen)
            self.actions[self.actualInputPosition] = action
            self.rewards[self.actualInputPosition] = reward
            self.isTerminal[self.actualInputPosition] = isTerminal
            self.actualInputPosition = (1 + self.actualInputPosition) % self.totalMemoryCapacity

    def _getState(self, index):
        assert(index >= 3)
        N, H, W = self.screens[0].size()
        state = torch.zeros(N * 4, H, W)
        for j in range(4):
            state[j, :, :] = self.screens[index - j]
        return state.to(DEVICE)

    def getSample(self, index):
        assert(index < self.actualMemoryLen - 1)
        state = self._getState(index)
        nextState = self._getState(index + 1)
        action = self.actions[index]
        reward = self.rewards[index]
        isTerminal = self.isTerminal[index]
        return (state, action, nextState, reward, isTerminal)
                
    def getBatch(self, batchSize = BATCH_SIZE, randomIndicies = None):
        if batchSize > self.actualMemoryLen:
            return None
        if randomIndicies == None:
            randomIndicies = np.random.choice(self.actualMemoryLen - 4, batchSize) + 3
            _, H, W = self.screens[0].size()
            statesBatch = torch.zeros(batchSize, 4, H, W).to(DEVICE)
            nextStatesBatch = torch.zeros(batchSize, 4, H, W).to(DEVICE)
            actionsBatch = torch.zeros(batchSize).type(torch.long).to(DEVICE)
            rewardsBatch = torch.zeros(batchSize).to(DEVICE)
            nonTerminalMask = torch.zeros(batchSize).type(torch.long).to(DEVICE)
            for j, index in enumerate(randomIndicies):
                state, action, nextstate, reward, isTerminal = self.getSample(index)
                statesBatch[j, :, :, :] = state
                nextStatesBatch[j, :, :, :] = nextstate
                actionsBatch[j] = action
                rewardsBatch[j] = reward
                nonTerminalMask[j] = 1 if isTerminal else 0
            return statesBatch, actionsBatch, nextStatesBatch, rewardsBatch, nonTerminalMask

    def __len__(self):
        return self.actualMemoryLen





def stateFromScreen(screen):
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=Image.CUBIC),
                        T.ToTensor()])
                        
    screen = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
    screen = screen[30:195,:]
    screen = np.ascontiguousarray(screen, dtype=np.uint8).reshape(screen.shape[0],screen.shape[1],1)
    return resize(screen).mul(255).type(torch.ByteTensor).to(DEVICE).detach()




EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

def epsilonGreedyChooser(normalAction, state, stepsDone):
    epsThreshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * stepsDone / EPS_DECAY)
    randomSample = random.random()
    if randomSample > epsThreshold:
        return normalAction(state).max(1)[1].view(1, 1)
    else:
        return ENVIRONMENT.action_space.sample()


def fillGameMemoryWithRandomTransitions(gameMemory):
    print("Preparing Dataset")
    progressBar = tqdm.tqdm(total = 100)
    while len(gameMemory) < REPLAY_MEMORY:
        progressBar.update((len(gameMemory) / REPLAY_MEMORY) * 100 )
        ENVIRONMENT.reset()
        isDone = False
        while not isDone:
            action = ENVIRONMENT.action_space.sample()
            screen, reward, isDone, info = ENVIRONMENT.step(action)
            ENVIRONMENT.render()
            gameMemory.pushScreenActionReward(screen, action, reward, isDone)
        print("dataset finished")








