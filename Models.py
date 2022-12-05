import numpy as np
class Solution():
    def __init__(self):
        self.BestCost = None
        self.BestSol = None
        self.CostCurve = None


class Particle():
    def __init__(self):
        self.Position = np.array([])
        self.Cost = np.array([])
        self.Velocity = np.array([])

        # to represent personal best
        self.BestPosition = np.array([])
        self.BestCost = np.array([])