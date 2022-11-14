import pandas as pd

def load_data():
    """
    Load data from CSV
    """
    df = pd.read_csv('Data.csv', header=None)
    global Eload = df[0]
    global G = df[1]
    global T = df[2]
    global Vw = df[3]


    """
    Input Data
    """
    # Type of system (1: included, 0=not included)
    PV=1
    WT=0
    DG=1
    Bat=1
    global Grid=0

    EM=0 # 0: LCOE, 1:LCOE+LEM

    Budget=200e3   # Limit On Total Capital Cost

    n = 25                  # life year of system (year)
    n_ir=0.0473             # Nominal discount rate
    e_ir=0.02               # Expected inflation rate
    ir=(n_ir-e_ir)/(1+e_ir) # real discount rate

    LPSP_max=0.011 # Maximum loss of power supply probability
    RE_min=0.75    # minimum Renewable Energy

    global Ppv_r=0.500  # PV module rated power (kW)
    global Pwt_r=1      # WT rated power (kW)
    global Cbt_r=1      # Battery rated Capacity (kWh)
    global Cdg_r=0.5    # Battery rated Capacity (kWh)

    # PV data
    # hourly_solar_radiation W
    global fpv=0.9       # the PV derating factor [%]
    global Tcof=0        # temperature coefficient
    global Tref=25       # temperature at standard test condition
    global Tnoct=45      # Nominal operating cell temperature
    global Gref = 1000   # 1000 W/m^2

    C_PV = 896       # Capital cost ($) per KW
    R_PV = 896       # Replacement Cost of PV modules Per KW
    MO_PV = 12       # O&M  cost ($/year/kw)
    L_PV=25          # Life time (year)
    n_PV=0.205       # Efficiency of PV module
    D_PV=0.01        # PV yearly degradation
    CE_PV=50         # Engineering cost of system per kW for first year
    RT_PV=ceil(n/L_PV)-1   # Replecement time

    # WT data
    global h_hub=17               # Hub height 
    global h0=43.6                # anemometer height
    nw=1                   # Efficiency
    global v_cut_out=25           # cut out speed
    global v_cut_in=2.5           # cut in speed
    global v_rated=9.5            # rated speed(m/s)
    global alfa_wind_turbine=0.14 # coefficient of friction ( 0.11 for extreme wind conditions, and 0.20 for normal wind conditions)

    C_WT = 1200      # Capital cost ($) per KW
    R_WT = 1200      # Replacement Cost of WT Per KW
    MO_WT = 40       # O&M  cost ($/year/kw)
    L_WT=20          # Life time (year)
    n_WT=0.30        # Efficiency of WT module
    D_WT=0.05        # PV yearly degradation
    RT_WT=ceil(n/L_WT)-1   # Replecement time

    # Diesel generator
    C_DG = 352       # Capital cost ($/KWh)
    global R_DG = 352       # Replacement Cost ($/kW)
    global MO_DG = 0.003    # O&M+ running cost ($/op.h)
    global TL_DG=131400     # Life time (h)
    n_DG=0.4         # Efficiency
    D_DG=0.05        # yearly degradation (%)
    global LR_DG=0.25       # Minimum Load Ratio (%)

    global C_fuel=1.24  # Fuel Cost ($/L)
    # Diesel Generator fuel curve
    global a=0.2730          # L/hr/kW output
    global b=0.0330          # L/hr/kW rated

    # Emissions produced by Disesl generator for each fuel in littre [L]	g/L
    CO2=2621.7
    CO = 16.34
    NOx = 6.6
    SO2 = 20

    # Battery data
    C_B = 360              # Capital cost ($/KWh)
    global R_B = 360              # Repalacement Cost ($/kW)
    MO_B=10                # Maintenance cost ($/kw.year)
    L_B=5                  # Life time (year)
    global SOC_min=0.2
    global SOC_max=1
    global SOC_initial=0.5
    D_B=0.05               # Degradation
    RT_B=ceil(n/L_B)-1     # Replecement time
    global Q_lifetime=8000        # kWh
    global self_discharge_rate=0  # Hourly self-discharge rate
    global alfa_battery=1         # is the storage's maximum charge rate [A/Ah]
    global c=0.403                # the storage capacity ratio [unitless] 
    global k=0.827                # the storage rate constant [h-1]
    global Imax=16.7              # the storage's maximum charge current [A]
    global Vnom=12                # the storage's nominal voltage [V] 
    global ef_bat=0.8             # storage DC-DC efficiency 
    # Inverter
    C_I = 788        # Capital cost ($/kW)
    R_I = 788        # Replacement cost ($/kW)
    MO_I =20         # O&M cost ($/kw.year)
    L_I=25           # Life time (year)
    global n_I=0.85         # Efficiency
    RT_I=ceil(n/L_I)-1 # Replecement time

    # Charger
    C_CH = 150  # Capital Cost ($)
    R_CH = 150  # Replacement Cost ($)
    MO_CH = 5   # O&M cost ($/year)
    L_CH=25     # Life time (year)
    RT_CH=ceil(n/L_CH)-1 # Replecement time

    # Price
    # TODO: not sure what's going on here

    # % Winter
    # Tp_w=[7:10 17:18];
    # Tm_w=11:16;
    # Toff_w=[1:6 19:24];

    # % Summer
    # Tp_s=11:16;
    # Tm_s=[7:10 17:18];
    # Toff_s=[1:6 19:24];

    P_peak=0.17
    P_mid=0.113
    P_offpeak=0.083

    # months
    # TODO: use constants for winter/summer months
    months = [0] * 12
    for i in range(4,10):
        months[i] = 1 # summer month
    day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # TODO: what is going on here
    # for m=1:12
        
    #     t_index=24*sum(Day(1:m-1))+1:24*sum(Day(1:m));
    #     nt=numel(t_index);
        
    #     if Month(m)==1     % for summer
    #         tp=Tp_s;
    #         tm=Tm_s;
    #         toff=Toff_s;
    #     else               % for winter
    #        tp=Tp_w;
    #        tm=Tm_w;
    #        toff=Toff_w;
    #     end
        
    #     Cbuy(t_index)=P_offpeak;
        
    #     for d=1:Day(m)
    # %Cbuy(t_index(toff)+24*(d-1))=P_offpeak;
    #     Cbuy(t_index(tp)+24*(d-1))=P_peak;
    #     Cbuy(t_index(tm)+24*(d-1))=P_mid;
    #     end
    # end

    Csell=0.1

    global Pbuy_max=ceil(1.2*max(Eload)) # kWh
    global Psell_max=Pbuy_max

    # Emissions produced by Grid generators (g/kW)
    E_CO2=1.43
    E_SO2=0.01
    E_NOx=0.39