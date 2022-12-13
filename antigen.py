#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:57:47 2022

@author: yutongli
"""

"""
This function is to describe how the antigen works in the system.
"""
class Antigen():
     def __init__(self, initial):
         self.initial = initial
     def antigen_rate(self, t, antigen, V_T):
     
         #All the calculated parameters are inside the list.
         V_e_P = antigen [0] # concentration of antigen in endosome
         V_e_p = antigen [1] # concentration of peptide in endosome
         A_e_Mp = antigen [2] #concentration of pMHC complexes in endosome
         A_s_Mp = antigen [3] #concentration of pMHC complexes on surface
         A_e_M = antigen [4] #concentration of MHC in endosome
         A_s_M = antigen [5] #concentration of MHC on surface
         V_T_P = antigen [6] #concentration of antigen in tumor
         
         
         
         
         #Add reactions here.   
         #Antigen deposition from dying cancer cells
         r_dep = k_dep = 
         #Free antigen degradation
         r_deg_P = k_xP_deg*V_T_P*V_T
         #Antigen uptake by mature antigen presenting cells
         r_uptake_T = k_up*V_T_APC*V_T_P*V_T 
         r_uptake_e = k_up*cell*V_T_P*V_e
         #Antigen degradation in APC endosomes'
         r_deg_anti = k_P_deg*V_e_P*V_e
         #Epitope degradation in APC endosomes
         r_deg_epi = k_p_deg*V_e_p*V_e
         #Antigen-MHC binding in endosome
         r_an_MHC_binding_en = k_on*V_e_p*A_e*MHC_T 
         #Antigen-MHC unbinding in endosome
         r_an_MHC_unbinding_en = k_on*A_e_Mp*A_e
         #Antigen-MHC unbinding on APC surface
         r_an_MHC_unbinding_sur = k_on*A_s_Mp*A_s
         #Antigen-MHC translocation
         r_an_MHC_trans = kout*A_e_Mp*A_e
         #MHC translocation
         r_MHC_trans = kout * MHC_T * A_e -kin * 1e-6 * A_s            
         
         
         #Calculate the ODE to get the antigens values
         dV_e_Pdt = (r_uptake_e - r_deg_anti)/V_e
         dV_e_pdt = (r_deg_anti - r_deg_epi - r_an_MHC_binding_en - r_an_MHC_unbinding_en)/V_e
         dA_e_Mpdt = (r_an_MHC_binding_en - r_an_MHC_unbinding_en - r_an_MHC_trans )/A_e
         dA_s_Mpdt = (r_an_MHC_trans - r_an_MHC_unbinding_sur)/A_s
         dA_e_Mdt = (r_an_MHC_unbinding_en - r_an_MHC_binding_en  - r_MHC_trans)/A_e
         dA_s_Mdt = (r_an_MHC_unbinding_sur + r_MHC_trans)/A_s
         dV_T_Pdt = (r_dep - r_deg_P  - r_uptake_T)/V_T        
         
         return [dV_e_Pdt, dV_e_pdt, dA_e_Mpdt, dA_s_Mpdt, dA_e_Mdt, dA_s_Mdt, dV_T_Pdt]
 
    def simulate(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution = solve_ivp(self.antigen_rate, [t_start, t_end], self.initial, intervals)
        return solution   
   
     
     def TCR(self,solution): 
         
         TCR_MHC_tot = 0.5 * (self.solution[3]/n_clones_tum + TCR_tot + k_TCR_off/k_TCR_on - sqrt(((self.solution[3]/n_clones_tum + TCR_tot + k_TCR_off/k_TCR_on)^2 - 4* self.solution[3]/n_clones_tum *TCR_tot))
         TCR_active = (k_TCR_off/(k_TCR_off+phi_TCR)) * ((k_TCR_p/(k_TCR_p+k_TCR_off))^N_TCR) * TCR_MHC_tot
         H_Ag = TCR_active/(TCR_active + p_50)
         
         return TCR_MHC_tot, TCR_active, H_Ag 
     
     
     
     



       
        
                  
        
        
      
        
