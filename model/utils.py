import torch
import os

def savecheckpoint(model, dir):
    torch.save(model.state_dict(), dir)

def loadcheckpoint(model, dir):
    if not os.path.exists(dir):
        raise KeyError("checkpoint is not exist")
    model.load_state_dict(torch.load(dir))