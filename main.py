import pandas as pd
from pulp import *

# Load data from CSV
data = pd.read_csv('input_data.csv', parse_dates=[0], index_col=0)  # parse the first column as datetime

# Define time periods
T = list(range(len(data)))  # Number of 5-minute intervals in the input data

# Define parameters
BESS_Cap = 250 # Your value here
BESS_Pmax = 100 # Your value here
SOC_min = 0 # Your value here
SOC_max = 250 # Your value here
N_cost_kWh = 0.2 # Your value here
N_cost_kW = 10 # Your value here
cycles_per_day = 1 # Your value here

# Get data from CSV
S_load = {t: data['site_load'][t] for t in T}
W_price = {t: data['wholesale_price'][t] / 1000 for t in T}  # Convert $/MWh to $/kWh
S_Pmax = {t: data['solar_generation'][t] for t in T}

# Initialize the problem
prob = LpProblem("Battery_Optimization", LpMinimize)

# Define variables
SOC = LpVariable.dicts("SOC", T, 0, BESS_Cap)
BESS_C = LpVariable.dicts("BESS_C", T, 0, BESS_Pmax)
BESS_D = LpVariable.dicts("BESS_D", T, 0, BESS_Pmax)
BESS_C_bin = LpVariable.dicts("BESS_C_bin", T, cat='Binary')
BESS_D_bin = LpVariable.dicts("BESS_D_bin", T, cat='Binary')
S_gen = LpVariable.dicts("S_gen", T, 0)

# Define objective function
prob += lpSum((BESS_C[t] - BESS_D[t] + S_load[t] - S_gen[t]) * (W_price[t] + N_cost_kWh) + N_cost_kW * (BESS_C[t] + S_load[t] - S_gen[t]) for t in T)

# Define constraints
for t in T:
    prob += SOC[t] <= BESS_Cap # BESS Capacity Constraint
    prob += SOC[t] >= SOC_min * BESS_Cap # Minimum SOC
    prob += SOC[t] <= SOC_max * BESS_Cap # Maximum SOC
    prob += BESS_C_bin[t] + BESS_D_bin[t] <= 1 # BESS Operation Constraint
    prob += BESS_C[t] <= BESS_Pmax * BESS_C_bin[t] # Linking Charge Power and Decision Variables
    prob += BESS_D[t] <= BESS_Pmax * BESS_D_bin[t] # Linking Discharge Power and Decision Variables
    prob += S_gen[t] <= S_Pmax[t] # Solar Power Output Constraint
    prob += BESS_D[t] + S_gen[t] >= S_load[t] # Site Load Must Be Met
    prob += BESS_C[t] + S_load[t] == BESS_D[t] + S_gen[t] # Power Balance

# For SOC Dynamics
for t in range(1, len(T)):
    prob += SOC[T[t]] == SOC[T[t-1]] + BESS_C[T[t]] - BESS_D[T[t]]

# Constraint for the number of cycles
prob += lpSum(BESS_C_bin[t] + BESS_D_bin[t] for t in T) / 2 <= cycles_per_day

# Solve the problem
prob.solve()
