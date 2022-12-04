import math

import numpy as np

import input_data

# https://mathesaurus.sourceforge.net/matlab-numpy.html


"""
Main method
"""
if __name__ == '__main__':
    input_data.load_data() # load input data

"""
Battery Model function
"""
def battery_model(
    Nbat, 
    Eb, 
    alfa_battery,
    c, 
    k, 
    Imax, 
    Vnom, 
    ef_bat, 
) -> (float, float):
    dt = 1         # the length of the time step [h]
    Q1 = c * Eb    # the available energy [kWh] in the storage at the beginning of the time step
    Q = Eb         # the total amount of energy [kWh] in the storage at the beginning of the time step
    Qmax = Nbat    # the total capacity of the storage bank [kWh]

    Pch_max1 = -(-k*c*Qmax+k*Q1*math.exp(-k*dt) + Q*k*c*(1-math.exp(-k*dt))) / (1-math.exp(-k*dt) + c*(k*dt-1+math.exp(-k*dt)))
    Pch_max2 = (1-math.exp(-alfa_battery*dt)) * (Qmax-Q) / dt
    Pch_max3 = Nbat * Imax * Vnom / 1000

    Pdch_max = (k*Q1*math.exp(-k*dt) + Q*k*c*(1-math.exp(-k*dt))) / (1-math.exp(-k*dt) + c*(k*dt-1+math.exp(-k*dt))) * sqrt(ef_bat)
    Pch_max = min([Pch_max1, Pch_max2, Pch_max3]) / sqrt(ef_bat)

    return Pdch_max, Pch_max

"""
Energy management 
"""
def energy_management(
    Ppv, Pwt, Eload, Cn_b, Nbat, Pn_DG, NT, Pinv_max, Cn_I, cc_gen, Cbw
):
    global SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy, a, LR_DG, C_fuel, Pbuy_max, Psell_max, self_discharge_rate, alfa_battery, c, k, Imax, Vnom, ef_bat

    Eb=zeros((1, NT))
    Pch=zeros((1, NT))
    Pdch=zeros((1, NT))
    Ech=zeros((1, NT))
    Edch=zeros((1, NT))
    Pdg=zeros((1, NT))
    Edump=zeros((1, NT))
    Ens=zeros((1, NT))
    Psell=zeros((1, NT))
    Pbuy=zeros((1, NT))
    Pinv=zeros((1, NT))
    Ebmax=SOC_max*Cn_B
    Ebmin=SPC_min*Cn_B
    Eb[0]=SOC_initial*Cn_B
    dt=1

    if Grid == 0:
        Pbuy_max=0
        Psell_max=0
    
    P_RE = Ppv + Pwt
    Pdg_min = 0.05*Pn_DG # LR_DG

    for t in range(NT):
        Pch_max, Pdch_max = battery_model(Cn_B, Nbat, Eb[t], alfa_battery, c, k, Imax, Vnom, ef_bat) # kW

        #  if PV+Pwt greater than load  (battery should charge)
        if P_RE[t] >= (Eload[t] / n_I) and Eload[t] <= Pinv_max: 
            # Battery charge power calculated based on surEloadus energy and battery empty  capacity
            Eb_e = (Ebmax - Eb[t]) / ef_bat
            Pch[t] = min(Eb_e, P_RE[t]-Eload[t]/n_I)
            
            # Battery maximum charge power limit
            Pch[t] = min(Pch[t], Pch_max)

            Psur_AC = n_I * (P_RE[t]-Pch[t]-Eload[t]) # surplus energy

            Psell[t] = min(Psur_AC, Psell_max)
            Psell[t] = min(max(0, Pinv_max-Eload[t]), Psell[t])

            Edump[t] = P_RE[t] - Pch[t] - (Eload[t]+Psell[t])/n_I
        
        # if load greater than PV+Pwt 
        else: 
            Edef_AC = Eload[t]-min(Pinv_max, n_I*P_RE[t])
            price_dg = cc_gen + a * C_fuel # DG cost ($/kWh)

            if (Cbuy[t] <= price_dg) and (price_dg <= Cbw): # Grid, DG , Bat : 1

                Pbuy[t] = min(Edef_AC, Pbuy_max)

                Pdg[t] = min(Edef_AC-Pbuy[t], Pn_DG)
                Pdg[t] = Pdg[t] * (Pdg[t] >= LR_DG*Pn_DG) + LR_DG*Pn_DG*(Pdg[t] < LR_DG*Pn_DG) * (Pdg[t] > Pdg_min) # TODO: what is this multiplication?

                Edef_AC=Eload[t]-Pdg[t]-Pbuy[t]-min(Pinv_max, n_I*P_RE[t])
                Edef_DC=Edef_AC/n_I*(Edef_AC>0) # TODO: what is this multiplication?
                Eb_e=(Eb[t]-Ebmin)*ef_bat
                Pdch[t] = min(Eb_e, Edef_DC)
                Pdch[t] = min(Pdch[t],Pdch_max)
                
                Esur_AC=-Edef_AC*(Edef_AC<0) # TODO: what is this multiplication?
                Pbuy[t]=Pbuy[t]-Esur_AC*(Grid==1)  # TODO: what is this multiplication?

            elif (Cbuy[t]<= Cbw) and (Cbw<price_dg): # Grid, Bat , DG : 2

                Pbuy[t]=min(Edef_AC,Pbuy_max)
                        
                Edef_DC=(Eload[t]-Pbuy[t])/n_I-P_RE[t]
                Eb_e=(Eb[t]-Ebmin)*ef_bat
                Pdch[t]= min(Eb_e,Edef_DC)
                Pdch[t]=min(Pdch[t],Pdch_max)
                
                Edef_AC=Eload[t]-Pbuy[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]))
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)  # TODO: what is this multiplication?
            
            elif (price_dg<Cbuy[t]) and (Cbuy[t]<=Cbw):  # DG, Grid , Bat :3
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min) # TODO: what is this multiplication?
                
                Pbuy[t]=max(0, min(Edef_AC-Pdg[t],Pbuy_max))
                Psell[t]=max(0, min(Pdg[t]-Edef_AC,Psell_max))
                
                Edef_DC=(Eload[t]-Pbuy[t]-Pdg[t])/n_I-P_RE[t]
                Edef_DC=Edef_DC*(Edef_DC>0) # TODO: what is this multiplication?
                Eb_e=(Eb[t]-Ebmin)*ef_bat
                Pdch[t] = min(Eb_e,Edef_DC)
                Pdch[t]=min(Pdch[t],Pdch_max)

            elif (price_dg<Cbw) and (Cbw<Cbuy[t]):  # DG, Bat , Grid :4
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min) # TODO: what is this multiplication?
                
                Edef_DC=(Eload[t]-Pdg[t])/n_I-P_RE[t]
                Edef_DC=Edef_DC*(Edef_DC>0) # TODO: what is this multiplication?
                Eb_e=(Eb[t]-Ebmin)*ef_bat
                Pdch[t]= min(Eb_e,Edef_DC)
                Pdch[t]= min(Pdch[t],Pdch_max)

                Edef_AC=Eload[t]-Pdg[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]))
                Pbuy[t]=max(0,  min(Edef_AC,Pbuy_max))
                Psell[t]=max(0, min(-Edef_AC,Psell_max))
                
            elif (Cbw<price_dg) and (price_dg<Cbuy[t]):  # Bat ,DG, Grid :5
                Edef_DC=Eload[t]/n_I-P_RE[t]
                Eb_e=(Eb[t]-Ebmin)*ef_bat
                Pdch[t]=min(Eb_e,Edef_DC)
                Pdch[t]=min(Pdch[t],Pdch_max)
                
                Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]))
                Pdg[t]=min(Edef_AC,Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)  # TODO: what is this multiplication?
                
                Pbuy[t]=max(0, min(Edef_AC-Pdg[t],Pbuy_max))
                Psell[t]=max(0, min(Pdg[t]-Edef_AC,Psell_max))
            else: # Bat , Grid , DG: 6
                
                Edef_DC=min(Pinv_max, Eload[t]/n_I)-P_RE[t]
                Eb_e=(Eb[t]-Ebmin)*ef_bat
                Pdch[t]=min(Eb_e,Edef_DC)*(Edef_DC>0)    # TODO: what is this multiplication?
                Pdch[t]=min(Pdch[t],Pdch_max) 
                
                Edef_AC=Eload[t]-min(Pinv_max, n_I*(P_RE[t]+Pdch[t]))
                Pbuy[t]= min(Edef_AC, Pbuy_max)
                
                Pdg[t]=min(Edef_AC-Pbuy[t],Pn_DG)
                Pdg[t]=Pdg[t]*(Pdg[t]>=LR_DG*Pn_DG)+LR_DG*Pn_DG*(Pdg[t]<LR_DG*Pn_DG)*(Pdg[t]>Pdg_min)  # TODO: what is this multiplication?
            
            Edef_DC=(Eload[t]+Psell[t]-Pdg[t]-Pbuy[t])/n_I-(P_RE[t]+Pdch[t]-Pch[t])

            if Edef_DC<0:
                Eb_e=(Ebmax-Eb[t])/ef_bat
                Pch[t]=min(Eb_e, Pch[t]-Edef_DC)
                Pch[t]=min(Pch[t],Pch_max)
            

            Esur=Eload[t]+Psell[t]-Pbuy[t]-Pdg[t]-min(Pinv_max, (P_RE[t]+Pdch[t]-Pch[t])*n_I) 
            Ens[t]=Esur*(Esur>0) # TODO: what is this multiplication?
            Edump[t]=-Esur*(Esur<0) # TODO: what is this multiplication?

        # Battery charging and discharging energy is determined based on charging and discharging power and the battery charge level is updated.
        Ech[t]=Pch[t]*dt
        Edch[t]=Pdch[t]*dt

        # index out of bounds error check
        if t < NT - 1:
            Eb[t+1]=(1-self_discharge_rate)*Eb[t]+ef_bat*Ech[t]-Edch[t]/ef_bat
    
    return Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell, Pinv
 
 
"""
Fitness function
"""
def fitness(X): # what is x? assume it is an array with 5 indices - might be better to explicitly define params
    global Eload, T, G, Vw
    global Ppv_r, Pwt_r, Cbt_r, Cdg_r
    global fpv, Tcof, Tref, Tnoct, Gref
    global h_hub, h0, alfa_wind_turbine, v_cut_out, v_cut_in, v_rated
    global R_B, Q_lifetime, ef_bat
    global a, b, C_fuel, R_DG, TL_DG, MO_DG, n
    global L_PV, R_PV, ir, L_WT, R_WT, L_DG, R_DG, L_B, R_B, L_I, R_I, L_CH, R_CH
    global MO_PV, MO_WT, MO_B, MO_I, MO_CH
    global RT_PV, RT_WT, RT_B, RT_I, RT_CH
    global CO2, NOx, SO2, E_CO2, E_SO2, E_NOx
    global Cbuy, Csell
    global EM, LPSP_max, RE_min, Budget

    NT = Eload.size # time step numbers

    Npv = round(X[0]) # PV number
    Nwt = round(X[1]) # WT number
    Nbat = round(X[2]) # Battery pack number
    N_DG = round(X[3]) # number of diesel generator
    Cn_I = X[4] # inverter capacity

    Pn_PV=Npv*Ppv_r     # PV Total Capacity
    Pn_WT=Nwt*Pwt_r     # WT Total Capacity
    Cn_B=Nbat*Cbt_r     # Battery Total Capacity
    Pn_DG=N_DG*Cdg_r    # Diesel Total Capacity

    # PV power calculation
    Tc = T+(((Tnoct-20)/800)*G) # Module Temprature
    Ppv = fpv*Pn_PV*(G/Gref)*(1+Tcof*(Tc-Tref)) # output power(kw)_hourly

    # Wind turbine Power Calculation
    v1=Vw # hourly wind speed

    v2 = ((h_hub / h0) ** alfa_wind_turbine) * v1 # v1 is the speed at a reference height;v2 is the speed at a hub height h2

    Pwt = zeros((1, 8761))
    for t in Pwt:
        if v2[t] < v_cut_in or v2[t] > v_cut_out:
            Pwt[t] = 0
        elif v_cut_in <= v2[t] and v2[t] < v_rated:
            Pwt[t] = v2[t]**3 * (Pwt_r / (v_rated**3 - v_cut_in)) - (v_cut_in**3 / (v_rated**3 - v_cut_in**3)) * Pwt_r
        elif v_rated <= v2[t] and v2[t] < v_cut_out:
            Pwt[t] = Pwt_r
        else:
            Pwt[t] = 0
        Pwt[t] = Pwt[t] * Nwt

    # Energy management
    # Battery wear cost
    if Cn_B > 0:
        Cbw = R_B * Cn_B / (Nbat * Q_lifetime * sqrt(ef_bat))
    else:
        Cbw = 0
    
    # DG fix cost
    cc_gen = b * Pn_DG * C_fuel + R_DG * Pn_DG / TL_DG + MO_DG

    (Eb, Pdg, Edump, Ens, Pch, Pdch, Pbuy, Psell) = energy_management(Ppv, Pwt, Eload, Cn_b, Nbat, Pn_DG, NT, Cn_I, cc_gen, Cbw) # these are global vars: SOC_max, SOC_min, SOC_initial, n_I, Grid, Cbuy,a,LR_DG,C_fuel,Pbuy_max,Psell_max,self_discharge_rate,alfa_battery,c,k,Imax,Vnom,ef_bat)

    q = (a * Pdg + b * Pn_DG) * (Pdg[Pdg > 0]) # fuel consumption of a diesel generator

    # installation and operation cost
    # total investment cost ($)
    I_Cost=C_PV*Pn_PV + C_WT*Pn_WT+ C_DG*Pn_DG+C_B*Cn_B+C_I*Cn_I +C_CH

    Top_DG = sum(Pdg[Pdg > 0]) + 1
    L_DG = TL_DG / Top_DG
    RT_DG = ceil(n / L_DG) - 1 

    # total replacement cost ($)
    RC_PV= zeros((1,n+1))
    RC_WT= zeros((1,n+1))
    RC_DG= zeros((1,n+1))
    RC_B = zeros((1,n+1))
    RC_I = zeros((1,n+1))
    RC_CH = zeros((1,n+1))

    RC_PV[L_PV+1:L_PV:n+1]= R_PV*Pn_PV / (1+ir) ** np.array([[1.001*L_PV], [L_PV], [n]])
    RC_WT[L_WT+1:L_WT:n+1]= R_WT*Pn_WT / (1+ir) ** np.array([[1.001*L_WT], [L_WT], [n]])
    RC_DG[L_DG+1:L_DG:n+1]= R_DG*Pn_DG / (1+ir) ** np.array([[1.001*L_DG], [L_DG], [n]])
    RC_B[L_B+1:L_B:n+1] = R_B*Cn_B / (1+ir) ** np.array([[1.001*L_B], [L_B], [n]])
    RC_I[L_I+1:L_I:n+1] = R_I*Cn_I / (1+ir) ** np.array([[1.001*L_I], [L_I], [n]])
    RC_CH[L_CH+1:L_CH:n+1] = R_CH / (1+ir) ** np.array([[1.001*L_CH], [L_CH], [n]])

    R_Cost=RC_PV+RC_WT+RC_DG+RC_B+RC_I+RC_CH

    # Total M&O Cost ($/year)
    MO_Cost=(MO_PV*Pn_PV + MO_WT*Pn_WT + MO_DG*sum(Pn_DG[Pn_DG>0])+ MO_B*Cn_B+ MO_I*Cn_I +MO_CH) / (1+ir) ** np.array([[1], [n]])

    # DG fuel Cost
    C_Fu= sum(C_fuel*q)/(1+ir) ** np.array([[1], [n]])

    # Salvage
    L_rem=(RT_PV+1)*L_PV-n
    S_PV=(R_PV*Pn_PV)*L_rem/L_PV * 1/(1+ir) ** n # PV
    L_rem=(RT_WT+1)*L_WT-n
    S_WT=(R_WT*Pn_WT)*L_rem/L_WT * 1/(1+ir) ** n # WT
    L_rem=(RT_DG+1)*L_DG-n
    S_DG=(R_DG*Pn_DG)*L_rem/L_DG * 1/(1+ir) ** n # DG
    L_rem=(RT_B +1)*L_B-n
    S_B =(R_B*Cn_B)*L_rem/L_B * 1/(1+ir) ** n
    L_rem=(RT_I +1)*L_I-n
    S_I =(R_I*Cn_I)*L_rem/L_I * 1/(1+ir) ** n
    L_rem=(RT_CH +1)*L_CH-n
    S_CH =(R_CH)*L_rem/L_CH * 1/(1+ir) ** n
    Salvage=S_PV+S_WT+S_DG+S_B+S_I+S_CH


    # Emissions produced by Disesl generator (g)
    DG_Emissions=sum(q*(CO2 + NOx + SO2))/1000 # total emissions (kg/year)
    Grid_Emissions= sum(Pbuy*(E_CO2+E_SO2+E_NOx))/1000 # total emissions (kg/year)

    Grid_Cost= (sum(Pbuy*Cbuy)-sum(Psell*Csell))* 1/(1+ir)** np.array([[1], [n]])

    # Capital recovery factor
    CRF=ir*(1+ir)**n/((1+ir)**n -1)

    # Totall Cost
    NPC=I_Cost+sum(R_Cost)+sum(MO_Cost)+sum(C_Fu)-Salvage+sum(Grid_Cost)
    Operating_Cost=CRF*(sum(R_Cost)+ sum(MO_Cost)+sum(C_Fu)-Salvage+sum(Grid_Cost))

    if sum(Eload-Ens) > 1:
        LCOE=CRF*NPC/sum(Eload-Ens+Psell)                # Levelized Cost of Energy ($/kWh)
        LEM=(DG_Emissions+Grid_Emissions)/sum(Eload-Ens) # Levelized Emissions(kg/kWh)
    else:
        LCOE = 100
        LEM = 100
    
    LPSP = sum(Ens) / sum(Eload)

    RE=1-sum(Pdg+Pbuy)/sum(Eload+Psell-Ens)
    RE.fillna(0)

    Z=LCOE+EM*LEM+10*(LPSP[LPSP>LPSP_max])+10*(RE[RE<RE_min])+100*(I_Cost[I_Cost>Budget])+100*max(0, LPSP-LPSP_max)+100*max(0, RE_min-RE)+100*max(0, I_Cost-Budget)

    return Z

def pso(
    data='Data.csv', # filename of data input or file object, default='Data.csv'
    PV=1, # Type of system (1: included, 0=not included)
    WT=0, # Type of system (1: included, 0=not included)
    DG=1, # Type of system (1: included, 0=not included)
    Bat=1, # Type of system (1: included, 0=not included)
):
    """
    Read data from csv
    """
    df = pd.read_csv(data, header=None)
    Eload = df[0]
    G = df[1]
    T = df[2]
    Vw = df[3]

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