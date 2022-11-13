import math

from pso import input_data

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
Fitness function
"""
def fitness(X): # what is x? assume it is an array with 5 indices - might be better to explicitly define params
    global Eload, T, G, Vw
    global Ppv_r, Pwt_r, Cbt_r, Cdg_r, 
    global fpv, Tcof, Tref, Tnoct, Gref
    global h_hub, h0, alfa_wind_turbine

    NT = Eload.size # time step numbers

    Npv = round(X[0]) # PV number
    Nxt = round(X[1]) # WT number
    Nbat = round(X[2]) # Battery pack number
    N_DG = round(X[3]) # number of diesel generator
    Cn_I = X[4] # inverter capacity

    Pn_PV=Npv*Ppv_r     # PV Total Capacity
    Pn_WT=Nwt*Pwt_r     # WT Total Capacity
    Cn_B=Nbat*Cbt_r     # Battery Total Capacity
    Pn_DG=N_DG*Cdg_r    # Diesel Total Capacity

    # PV power calculation
    Tc = T+(((Tnoct-20)/800)*G) # Module Temprature

    # TODO: what is the dot operator
    # https://www.mathworks.com/help/matlab/matlab_prog/matlab-operators-and-special-characters.html
    Ppv = fpv*Pn_PV.*(G/Gref).*(1+Tcof.*(Tc-Tref)) # output power(kw)_hourly

    # Wind turbine Power Calculation
    v1=Vw # hourly wind speed

    # TODO: matrix power
    v2=((h_hub/h0)^(alfa_wind_turbine))*v1 # v1 is the speed at a reference height;v2 is the speed at a hub height h2
