from copy import deepcopy as copy

class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.virtue = 0
        self.prevLoss = float('inf')
        self.readySetStop = False
        self.theOneModelToRuleThemAll = None 
    
    def __call__(self, valLoss, model):
        if self.readySetStop:
            print("Why are you still training???  Overfitting...")
        elif (valLoss < self.prevLoss):
            self.virtue = 0
            self.prevLoss = valLoss
            self.theOneModelToRuleThemAll = copy(model)
        else:
            self.prevLoss = valLoss
            self.virtue += 1 
            if self.virtue >= self.patience:
                self.readySetStop = True
               
        return self.readySetStop, self.theOneModelToRuleThemAll