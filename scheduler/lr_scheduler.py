class SimpleScheduler(object):
    
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        
    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()
        self.optimizer.step()
    
    def get_lr(self):
        return self.lr