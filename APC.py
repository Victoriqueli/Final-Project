"""
Created on Fri Dec  2 00:51:41 2022

@author: yutongli
"""
'''https://github.com/popellab/QSPIO-TNBC/tree/main/parameters'''
"""

"""


import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.integrate import solve_ivp


RC = {
     #Those parameters are for the APC module. 
    "APC0_T" : 4.0e5
    "k_APC_mat" : 1.5 #(1/day)
    "c50" : 1.0e-9 #molarity
    "k_APC_death" : 0.01 #(1/day)
    "APC0_LN" : 1.2e6 #(cell/milliliter)
    "nLNs" : 17 # dimensionless
    "D_LN" : 5 # millimeter
    "k_APC_mig" : 4 #(1/day)
    "k_mAPC_death" : 0.02 #(1/day)
    "k_c" : 2.0 #(1/day)
    "c0" : 1.0e-9 #(molarity)
    "R_Tcell" : 
    "DAMPs" : 1.34e-14 #(mole/cell)
    
    #Those part are for the antigens.
    "k_dep" : 
    "k_xP_deg" : 2.0 #Rate of Extracellular Antigen Degradation(1/day)
    "k_up" : 14.4 #Rate of Antigen Uptake (1/day/cell)
    "cell" : 1 #Define Cell Dimension (cell)
    "k_P_deg" : 17.28 #Rate of Endosomal Antigen Degradation (1/day)
    "k_p_deg" : 144.0 #Rate of Endosomal Epitope Degradation (1/day)
    "k_on" : 1.44e5 # Rate of Antigen Binding (1/day/molarity)
    "A_s" : 900.0 #Endosomal Surface Area(micrometer^2)
    "kout" : 28.8 #Rate of MHC Externalization (1/day)
    "N_MHC" : 1  # Number of MHC Molecule Types (dimensionless)
    "n_MHC_T" : 2e6 #Total Amount of MHC (molecule)
    "kin" : 14.4 #Rate of MHC Internalization (1/day)
    "N_endo" : 10 #Number of Endosomal Vesicles per Cell (dimensionless)
    "V_endo" : 4.0e-17 # Endosomal Volume (liter)
    "A_endo" : 1.5 #Endosomal Surface Area (micrometer^2)
    "A_APC" : 900.0 # Surface Area of APC Cells (micrometer^2)
    "N_p_50" : 1e-3 # Number of Epitope Molecules for Half-Maximal T Cell Activation (molecule)
    
    "A_syn" : 37.8 # Surface area of the synapse (micrometer^2)
    "TCR_tot_abs" : 15708 # Number of TCR molecules on naive T cells (molecule)
    "D_cell" : 17 # Cancer Cell Diameter (micrometer)
    "D_Tcell": 6.94 #T Cell Diameter (micrometer)
    "TCR_tot_abs" : 15708 # Number of TCR molecules on naive T cells (molecule)
    "k_TCR_p" : 1 # Rate of modification of TCRs (1/second)
    "k_TCR_off" : 1 #Unbinding rate of ag/MHC to TCR (1/second)
    "k_TCR_on" : 1e-0 #binding rate of ag/MHC to TCR (1/(second*molecule/micrometer^2))
    "phi_TCR" : 0.09 #Rate of modification of TCR that leads to non-signaling (1/second)
    "N_TCR" : 10 # Number of intermediate steps (dimensionless)
    
    #Those part are for the Teff cells.
    "Q_nCD8_thym" : 3.5e7 #cell/day
    "nCD8_div" : 1.11e6 #dimensionless
    "k_nCD8_pro" : 3.2e8 #cell/day
    "K_nT_pro" : 1e9 #cell
    "k_nT_death" : 0.002 #1/day
    "q_nCD8_P_in" : #zhi jie zai apc xie le
    "q_nCD8_P_out" : 5.1 #1/day
    "q_nCD8_LN_in" : 0.076 #1/day
    "q_nCD8_LN_out" : 1.8 #1/day
    
    "k_nT_mig" : 4.2e-13 # Naive T Rate of Transmigration (1/minute/cell)
    "rho_adh" : 5e8 # T cell Adhesion Density (cell/centimeter^3)
    "gamma_P" : 0.014 # Peripheral Vascular Volume Fractions (dimensionless)
    "V_P" : 60 # Peripheral Compartment Volume (liter)
    "k_nCD8_act": 23 #Maximum Rate of CD8+ T Cell Activation by mAPCs(1/day)
    "n_clones_tum" : 63 #Tumor-specific T cell clone number (TMB) (Dimensionless)
    
    "kc_growth": 0.0072 #(1/day)
    "kc_death": 0.0001 #(1/day)
    "C_max": 2.7e4 # (cell)
    "C_total": 
 
    "k_CD8_pro": 1.0 #Rate of CD8+ T Cell Proliferation (1/day)
    "k_CD8_death":0.01 #Rate of CD8 T Cell Decay (1/day)
    "k_cell_clear": 0.1 #Dead Cell Clearance Rate (1/day)
    "Kc_rec": 2.02e7 #Half-Maximal cancer cell number for T cell recruitment (cell^2)
    "k_Treg" : 0.1 #Rate of T Cell Death by Tregs (1/day)
    "k_Tcell" : 0.1 # Rate of T Cell Exhaustion by Cancer Cells (1/day)
    "k_CD8_mig" : 5.8e-12 # Activated CD8 Rate of Transmigration(1/minute/cell)
    "rho_adh" : 5e8 # T cell Adhesion Density (cell/centimeter^3)
    "gamma_P" : 0.014 # Peripheral Vascular Volume Fractions (dimensionless)
    "V_P" : 60 # Peripheral Compartment Volume (liter)
    "q_CD8_P_out" : 24 # Activated CD8+ T Cell Transport P->C (1/day)
    "gamma_T" : 0.02 # Tumour Vascular Volume Fractions (dimensionless)
    "q_CD8_LN_out" : 24.0 #Activated CD8+ T Cell Transport LN->C (1/day)
    "k_IL2_deg" : 0.2 # Degradation Rate (1/minute)
    "k_IL2_cons" :  6.0e-6 # Maximum Consumption Rate by T Cells (nanomole/cell/hour)
    "IL2_50" : 0.32 # IL2 Concentration for Half-Maximal T Cell Proliferation (nanomolarity)
    "N0" : 2 # Baseline Number of Activated T Cell Generations (dimensionless)
    "N_costim" : 3 #Baseline Number of Activated T Cell Generations for co-stimulation (dimensionless)
    "N_IL2_CD8" : 11 # Additional Number of Activated CD8+ T Cell Generations Due to IL2 (dimensionless)  
    "Ve_T": 0.37 #Tumor Cell Volume Fraction(dimensionless)
    "k_cell_clear": 0.1 #Dead Cell Clearance Rate (1/day)
    "PD1": 3.1e3*20*.45 #PD1 Expression on T Cells (molecule)
    "PD1_50": 6 #PD1/PDL1 Concentration for Half-Maximal T Cell Killing (molecule/micrometer^2)
    "n_PD1": 2 #Hill Coefficient for PD1/PDL1 (dimensionless)
    "kon_PD1_PDL1": 0.18 #PD1-PDL1 kon(1/(micromolarity*second))
    "kd_PD1_PDL1": 8.2 #PD1-PDL1 kd (micromolarity)
    "k_out_PDL1": 5e4 #Expression rate of PDL1 on tumor cells (molecule/day)
    "IFNg_50_ind": 2.96 #Half-Maximal IFNg level for PD-L1 induction (picomolarity)
    "k_in_PDL1": 1 #Degradation rate of PDL1 on tumor cells(1/day)
    "PDL1_base": 
    "PDL1_total":
        
     "n_sites_APC": 10 #dimensionless
     "cell": 1
     "n_T0_clone" :log(63) ##number of neoantigen clones for naive CD4 T cell
     "n_T1_clones" : log(63) ##number of neoantigen clones for naive CD8 T cell     
    
    
}
    
RC['V_LN'] =  "nLNs" *4/3*pi*("D_LN"/2)^3 # milliliter
RC["V_e"] = "V_endo" * "N_endo" #Endosomal Volume
RC["MHC_T"] = "n_MHC_T" /("A_endo" * "N_endo" + "A_s") #Total Amount of MHC per Area
RC["A_e"] = 'A_endo' * 'N_endo' #Endosomal Surface Area (micrometer^2)
RC["p_50"] = "N_p_50" / "A_syn" # TCR-pMHC Concentration for Half-Maximal T Cell Activation (molecule/micrometer^2)]
RC["A_cell"] = 4*pi*("D_cell"/2)^2 #Surface Area of Cancer Cells (micrometer^2)
RC["A_Tcell"] = 4*pi*("D_Tcell"/2)^2 #Surface Area of T Cells (micrometer^2)
RC["TCR_tot"] = "TCR_tot_abs"/"A_Tcell" # TCR molecules density on naive T cells (molecule/micrometer^2)
RC["q_nCD8_P_in"] = "k_nT_mig" * "rho_adh" * "gamma_P" * "V_P" # Naive CD8+ T Cell Transport C->P (1/minute)
RC["q_CD8_P_in"] = "k_CD8_mig" * "rho_adh" * "gamma_P" * "V_P""
RC["q_CD8_T_in"] = "k_CD8_mig" * "rho_adh" * "gamma_T"
RC["vol_cell"] = 4/3*pi*("D_cell"/2)^3/"cell" #Volume of a cancer cell calculated based on cancer cell diameter
RC["vol_Tcell"] = 4/3*pi*("D_Tcell"/2)^3/"cell" #Volume of a T cell calculated based on the average T cell diameter
RC["koff_PD1_PDL1"] = "kon_PD1_PDL1" * "kd_PD1_PDL1" 

'''
This class contains the number of Antigen-Presneting Cells 
and the Hill Equations between the APCs and T cells.
'''

class APC():
    
    def __init__(self, initial):
        self.initial = initial 
        
    def inAPC(self,t,APC,RTcell):
        #All the parameters that we needed are put inside the APC matrix.
        APC_tumor = APC[0]
        APC_LN = APC[1]
        mAPC_tumor = APC[2]
        mAPC_LN = APC[3]
        c = APC[4]
        
        
        #Those are parameters for the equations to calculate.
        
            
        #APC recruitment/death in the tumour
        #0 -> APC_tumor 
        r_APC_death_tumor = RC["k_APC_death"]*(RC["APC0_T"*V_T-APC_tumor)
        #APC maturation in the tumour
        #APC_tumor ->  mAPC_tumor
        r_APC_maturation_tumor = RC["k_APC_mat"]*c/(c+RC["c50"])*APC_tumor
        #mAPC death in the tumour
        # mAPC_tumor -> 0
        r_mAPC_death_tumor = RC["k_mAPC_death"] * mAPC_tumor
        #APC recruitment/death in LN
        #0 -> APC_LN
        r_APC_death_LN = RC["k_APC_death"]*(RC["APC0_LN"]*RC["V_LN"]-APC_LN)
        #APC migration to the lymph node
        #mAPC_tumor -> mAPC_LN
        r_mAPC_migration_LN = RC["k_APC_mig"]*mAPC_tumor
        #mAPC death in the lymph node'
        #mAPC_LN -> 0
        r_mAPC_death_LN = RC["k_mAPC_death"]*mAPC_LN
        #Baseline cytokine secretion/degradation
        #0 -> c
        r_cytokines_baseline_secreation = RC["k_c"]*(RC["c0"]-c)
        #Cytokine release in response to the tumour
        #0 -> c
        r_cytokines_tumor_release = R_Tcell*RC["DAMPs"]
        
        #Calculate the ODE to get the APCs values
        dAPC_tumordt = r_APC_death_tumor - r_APC_maturation_tumor
        dAPC_LNdt = r_APC_death_LN 
        dmAPC_tumordt = r_APC_maturation_tumor - r_mAPC_migration_LN - r_mAPC_death_tumor
        dmAPC_LNdt = r_mAPC_migration_LN - r_mAPC_death_LN
        dcdt = r_cytokines_baseline_secreation + r_cytokines_tumor_release
        
        return [dAPC_tumordt, dAPC_LNdt, dmAPC_tumordt, dmAPC_LNdt, dcdt]
    
        
        
    def simulate(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution = solve_ivp(self.inAPC, [t_start, t_end], self.initial, intervals)
        return solution         

    def Hill_equation(self, mAPC_LN):
        
        
        #Hill equation between APC and T cell
        H_APC = RC["n_sites_APC"]*self.solution[3]/(RC["n_sites_APC"]*self.solution[3]+V_LN.nT0*RC["n_T0_clones"]+RC["cell"]
        #Hill equation between mAPC and T cell
        H_mAPC = n_sites_APC*mAPC_LN/(n_sites_APC*mAPC_LN+V_LN.nT1*n_T1_clones+cell
        #Hill equation between APC and helper T cell
        H_CD28_APC = n_sites_APC*mAPC_LN/(n_sites_APC*mAPC_LN+V_LN.nT0*n_T1_clones+cell
                                            
        return H_APC, H_mAPC, H_CD28_APC
    
   
   
