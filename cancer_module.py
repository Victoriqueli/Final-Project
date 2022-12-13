#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:13:05 2022

@author: yutongli
"""

"""
This is a function working on cancer size module 
"""

import math

class Cancer():
    def __init__(self, initials):
        self.initials = initials
        
    def parts(self, cancer_size): 
        Cx = cancer_size[0]
        T1_exh = cancer_size[1]
        Th_exh = cancer_size[2]
        
        
        #Those are the reactions to describe the parameters used to calculate the tumor size.
        #Clearance of dead cancer cells from tumor.
        r_dead_cancer_tumor = RC["k_cell_clear"] * Cx
        #Clearance of exhausted CD8 T cells from tumor
        r_dead_CD8_T_tumor = RC["k_cell_clear"]*T1_exh
        #Clearance of exhausted CD4 T cells from tumor
        r_dead_CD4_T_tumor = RC["k_cell_clear"]*Th_exh
        
        #This is to write an ODE to express Cx, T1_exh and Th_exh
        dCxdt = - r_dead_cancer_tumor 
        dT1_exhdt = - r_dead_CD8_T_tumor
        dTh_exhdt = -  r_dead_CD4_T_tumor
        
        return [dCxdt, dT1_exhdt, dTh_exhdt]
    
    def simulate(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution = solve_ivp(self.parts, [t_start, t_end], self.initial, intervals)
        return solution 
    
    def size(self, solution):      
        #This describe the size of the tumor cells 
        V_T = ((solution[0] + C_total)*RC["vol_cell"]+(solution[1]+solution[2] + T_total)*RC["vol_Tcell"])/RC["Ve_T"]
        return V_T
    
        


class Checkpoint():
    def __init__(self, initials):
        self.initials = initials
        
    def PDL1(self, PDL):
        PD1_PDL1 = PDL[0]
        PDL1 = PDL[1] 
        
        #Those are the reactions to describe the parameters used to calculate the number of PDL1-PD1 binding.
        #Binding and unbinding of PD1 PDL1 in synapse
        r_PD1_PDL1 = RC["kon_PD1_PDL1"]*PD1*PDL1 - koff_PD1_PDL1* PD1_PDL1
        #Translocation of PDL1 between cell surface and cytoplasm
        r_PDL1_trans_out = k_out_PDL1*V_T.IFNg/(V_T.IFNg+IFNg_50_ind) 
        #Translocation of PDL1 between cell surface and cytoplasm
        r_PDL1_trans_in = k_in_PDL1*(PDL1_base/A_cell - PDL1_total) 
       
        
        #Create the ode to calculate PDL1 and PD1_PDL1
        dPD1_PDL1dt = r_PD1_PDL1
        dPDL1dt = r_PDL1_trans_out + r_PDL1_trans_in - r_PD1_PDL1
        return [dPD1_PDL1dt, dPDL1dt]
    
    def simulate(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution = solve_ivp(self.PDL1, [t_start, t_end], self.initial, intervals)
        return solution    
    
        
    def Hill_equation(self, PD1_PDL1):
        
        #Hill equation between PD1 and cancer (PDL1)
        H_PD1_C1 = (PD1_PDL1 /PD1_50)^n_PD1/(((PD1_PDL1)/PD1_50)^n_PD1 + 1)
        return H_PD1_C1
        