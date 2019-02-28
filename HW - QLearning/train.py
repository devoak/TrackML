#----------ESSENTIAL_IMPORTS-----------------
from config import *
from model import *
from utils import *
#--------------------------------------------




def mainCycle():
    """
        Initialising two nets
        first one - key net, which we are training
        second one - net, that we use to estimate Q-value-function
            for next states for Bellman Equation
    """
    #--------------------------------------------
    keyNet = QModel().to(DEVICE)
    helperNet = QModel().to(DEVICE)
    helperNet.load_state_dict(keyNet.state_dict())
    helperNet.eval()
    #--------------------------------------------
    optimizer = optim.Adam(keyNet.parameters(), lr = 1e-4)
    gameMemory = GameMemory()
    ENVIRONMENT.render()

    
    fillGameMemoryWithRandomTransitions(gameMemory)
    stepsDone = 0
    normalAction = lambda state: keyNet(state).max(1)[1].view(1, 1)
    stateHolder = OneStateHolder()
    
    
    print("started learning")
    for e in tqdm.tqdm(range(AMOUNT_OF_EPISODES)):
        totalDisqountedReward = 0
        ENVIRONMENT.reset()
        stateHolder.initWithFirstScreens()
        isDone = False
        while not isDone:
            #performing action choosing according to epsilon greedy rule
            ENVIRONMENT.render()
            action = epsilonGreedyChooser(normalAction, stateHolder.getState().unsqueeze(0), stepsDone)
            stepsDone += 1
            screen, reward, isDone, info = ENVIRONMENT.step(action)
            stateHolder.pushScreen(screen)
            gameMemory.pushScreenActionReward(screen, action, reward, isDone)
            
            totalDisqountedReward = reward + totalDisqountedReward * DISCOUNT_FACTOR
            
            #performing neural network training on our replayMemory
            statesBatch, actionsBatch, nextStatesBatch, rewardsBatch, terminalMask = gameMemory.getBatch()
            
            currentQValues = keyNet(statesBatch).gather(1, actionsBatch.unsqueeze(1))
            nextQValues = torch.zeros(BATCH_SIZE, device = DEVICE)
            nextQValues = helperNet(nextStatesBatch).max(1)[0].detach()
            nextQValues[terminalMask == 1] = 0
            expectedQValues = rewardsBatch + nextQValues * DISCOUNT_FACTOR
            expectedQValues = torch.tensor(expectedQValues).unsqueeze(1).to(DEVICE)
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)
            optimizer.zero_grad()
            loss.backward()
        
        if e % HELPER_UPDATE == 0:
            helperNet.load_state_dict(keyNet.state_dict())

        print("totalDisqountedReward = %f" % (totalDisqountedReward))




            










if __name__ == "__main__":
    mainCycle()
    ENVIRONMENT.close()
