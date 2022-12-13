#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:51:57 2022

@author: yutongli
"""

class Teff():
    def __init__(self, initials):
        self.initial = initial
        
    def naive_T(self, nT): 
        
        #Those calculated parameters are saved in the matrix.
        V_C_nT = nT[0]
        V_P_nT = nT[1]
        V_LN_nT = nT[2]
        
        #Those are the parameters needed.
        Q_nCD8_thym = 
        nCD8_div = 
        k_nCD8_pro = 
        K_nT_pro = 
        k_nT_death = 
        q_nCD8_P_in = 
        q_nCD8_P_out = 
        q_nCD8_LN_in = 
        q_nCD8_LN_out = 
        
        
        
        
        
        #Those are the reactions for T cell.
        #Those part is for naive T cell dynamics
        #Thymic output of naive T cell to blood
        r_nT_thymic = Q_nCD8_thym/nCD8_div
        #Naive T cell proliferation in the peripheral compartment
        r_nT_pro_pheripheral = k_nCD8_pro/nCD8_div*V_P_nT/(K_nT_pro/nCD8_div+V_P_nT
        #Naive T cell proliferation in the TDLN compartment
        r_nT_pro_TDLN = k_nCD8_pro/nCD8_div*V_LN_nT/(K_nT_pro/nCD8_div+V_LN_nT)                                        
        #Naive T cell death in the peripheral compartment
        r_nT_death_pheripheral = k_nT_death*V_P_nT
        #Naive T cell death in the central compartment
        r_nT_death_central = k_nT_death*V_C_nT
        #Naive T cell death in the TDLN compartment
        r_nT_death_TDLN = k_nT_death*V_LN_nT
        #Naive T cell entry into the peripheral compartment
        r_nT_entry_peripheral = q_nCD8_P_in*V_C_nT
        #Naive T cell exit from the peripheral compartment
        r_nT_exit_peripheral = q_nCD8_P_out*V_P_nT
        #Naive T cell entry into the lymph node
        r_nT_entry_TDLN = q_nCD8_LN_in*V_C_nT
        #Naive T cell exit from the lymph node
        r_nT_exit_TDLN = q_nCD8_LN_out*V_LN_nT
        
        
        
        #Calculated the ODEs for the naive Tcells.
        dV_C_nTdt = r_nT_thymic - r_nT_entry_peripheral + r_nT_exit_peripheral - r_nT_entry_TDLN + r_nT_exit_TDLN - r_nT_death_central
        dV_P_nTdt = r_nT_pro_pheripheral + r_nT_entry_peripheral - r_nT_exit_peripheral - r_nT_death_pheripheral
        dV_LN_nTdt = r_nT_pro_TDLN + r_nT_entry_peripheral - r_nT_exit_peripheral - r_nT_death_TDLN - 
    
        
    
        
                                        
                                                    
        
        
    

    def Get_Diameter(self):
        V_T = ((C_x+C_total)*vol_cell+(T1_exh+Th_exh+T_total)*vol_Tcell)/Ve_T