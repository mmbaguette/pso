import numpy as np

from Input_Data import Data
from EMS import energy_management
from Battery_Model import battery_model
from Fitness import fitness
from Models import Solution, Particle

# https://mathesaurus.sourceforge.net/matlab-numpy.html


"""
Main method
"""
if __name__ == '__main__':
    pso()

"""
PSO function
"""
def pso(
    data='Data.csv', # filename of data input or file object, default='Data.csv'
    **kwargs
):
    """
    Read data from csv
    """
    df = pd.read_csv(data, header=None)
    Eload = df[0]
    G = df[1]
    T = df[2]
    Vw = df[3]

    inputs = Data(Eload, G, T, Vw) # load input data
    inputs.set_user_data(kwargs) # set user defined values

    nVar = 5                # number of decision variables
    VarSize = [0, nVar]     # size of decision variables matrix

    # Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
    VarMin = np.array([0,0,0,0,0]) # Lower bound of variables
    VarMax = np.array([100,100,60,10,20]) # Upper bound of variables

    VarMin = VarMin * [PV, WT, Bat, DG, 1]
    VarMax = VarMax * [PV, WT, Bat, DG, 1]

    # PSO parameters
    MaxIt = 100 # Max number of iterations
    nPop = 50 # Population size (swarm size)
    w = 1 # Inertia weight
    wdamp = 0.99 # Inertia weight damping ratio
    c1 = 2 # Personal learning coefficient
    c2 = 2 # Global learning coefficient

    # Velocity limits
    VelMax = 0.3 * (VarMax - VarMin)
    VelMin = -VelMax

    Run_Time = 1

    solution_particle = Solution()
    Sol = np.kron(ones((Run_Time, 1)), solution_particle)

    for tt in range(Run_Time):
        w = 1 # intertia weight 

        # initialization
        empty_particle = Particle()
        particle = np.kron(ones((nPop, 1)), empty_particle)

        GlobalBest = {
            "Cost": float('inf'),
            "Position": None
        }

        for i in range(nPop):
            # initialize position
            particle[i].Position = np.random.uniform(VarMin, VarMax, VarSize) # TODO: discrete uniform distribution (not continuous)
            
            # initialize velocity
            particle[i].Velocity = zeros(VarSize)
            
            # evaluation
            particle[i].Cost = fitness(particle[i].Position)
            
            # update personal best
            particle[i].BestPosition = particle[i].Position
            particle[i].BestCost = particle[i].Cost

            # Update global best
            if particle[i].BestCost < GlobalBestCost:
                GlobalBest["Cost"] = particle[i].BestCost
                GlobalBest["Position"] = particle[i].BestPosition
    
        BestCost = zeros((MaxIt, 1))
        MeanCost = zeros((MaxIt, 1))

        # PSO main loop
        for it in range(MaxIt):
            for i in range(nPop):

                # update velocity
                particle[i].Velocity = w * particle[i].Velocity + c1 * random.uniform(0,1,(VarSize,VarSize)) * (particle[i].BestPosition - particle[i].Position) + c2 * random.uniform(0,1,(VarSize,VarSize)) * (GlobalBest["Position"]-particle[i].Position)

                # apply velocity limits
                particle[i].Velocity = max(particle[i].Velocity, VelMin)
                particle[i].Velocity = min(particle[i].Velocity, VelMax)

                # update position
                particle[i].Position = particle[i].Position + particle[i].Velocity

                IsOutside = particle[i].Position < VarMin or particle[i].Position > VarMax
                # particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside) #TODO: What is this evaluation

                # Apply position limits
                particle[i].Position = max(particle[i].Position, VarMin)
                particle[i].Position = min(particle[i].Position, VarMax)

                # evaluation
                particle[i].Cost = fitness(particle[i].Position)

                # update personal best
                if particle[i].Cost < particle[i].BestCost:
                    particle[i].BestPosition = particle[i].Position
                    particle[i].BestCost = particle[i].Cost

                    # update global best
                    if particle[i].BestCost < GlobalBest["Cost"]:
                        GlobalBest["Position"] = particle[i].BestPosition
                        GlobalBest["Cost"] = particle[i].BestCost
        
            BestCost[it] = GlobalBest["Cost"]
            temp = 0
            for j in range(nPop):
                temp = temp + particle[j].BestCost
            MeanCost[it] = temp / nPop

            print("Run time = ", tt)
            print("Iteration = ", it)
            print("Best Cost = ", BestCost[it])
            print("Mean Cost = ", MeanCost[it])

            w = w*wdamp
    
        Sol[tt].BestCost = GlobalBest["Cost"]
        Sol[tt].BestSol = GlobalBest["Position"]
        Sol[tt].CostCurve = BestCost


    Best = [Sol.BestCost] # TODO: what is this
    # [~,index]=min(Best) # TODO: what is this, what is index?
    X=Sol[index].BestSol