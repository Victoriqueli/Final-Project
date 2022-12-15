"""
Test to show how to use solve_ivp
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.integrate import solve_ivp

def exponential_decay(t, y): 
    return -0.5 * y

sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])

print(sol.t)
print(sol.y)



  

  
    
   
   
