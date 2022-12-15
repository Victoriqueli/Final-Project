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
    "DAMPs" : 1.34e-14 #(mole/cell)
    
    #Those part are for the antigens.
    "k_dep" : 0.0034 #1/day    
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
    
 
    "k_CD8_pro": 1.0 #Rate of CD8+ T Cell Proliferation (1/day)
    "k_CD8_death":0.01 #Rate of CD8 T Cell Decay (1/day)
    "k_cell_clear": 0.1 #Dead Cell Clearance Rate (1/day)
    "Kc_rec": 2.02e7 #Half-Maximal cancer cell number for T cell recruitment (cell^2)
    "k_Treg" : 0.1 #Rate of T Cell Death by Tregs (1/day)
    "k_Tcell" : 0.1 # Rate of T Cell Exhaustion by Cancer Cells (1/day)
    "k_CD8_mig" : 5.8e-12 # Activated CD8 Rate of Transmigration(1/minute/cell)
    
    
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
    "PD1": 3.1e3*20*.45 #PD1 Expression on T Cells (molecule)
    "PD1_50": 6 #PD1/PDL1 Concentration for Half-Maximal T Cell Killing (molecule/micrometer^2)
    "n_PD1": 2 #Hill Coefficient for PD1/PDL1 (dimensionless)
    "kon_PD1_PDL1": 0.18 #PD1-PDL1 kon(1/(micromolarity*second))
    "kd_PD1_PDL1": 8.2 #PD1-PDL1 kd (micromolarity)
    "k_out_PDL1": 5e4 #Expression rate of PDL1 on tumor cells (molecule/day)
    "IFNg_50_ind": 2.96 #Half-Maximal IFNg level for PD-L1 induction (picomolarity)
    "k_in_PDL1": 1 #Degradation rate of PDL1 on tumor cells(1/day)
    "PDL1_base": 120000 #molecule    
    "PDL1_total": 83700 #molecule 
    "k_C1_death": 0.0001 #1/day       
    "k_C_nabp":  0.06 #1/hour    
    "IC50_nabp": 92 #nanomolarity     
    
     "n_sites_APC": 10 #dimensionless
     "cell": 1
     "n_T0_clones" :log(63) ##number of neoantigen clones for naive CD4 T cell
     "n_T1_clones" : log(63) ##number of neoantigen clones for naive CD8 T cell     
     "Kc_nabp": 8e+07 #cell 
     "K_T_C": 1.2 #dimensionless 
     "R_Tcell": 1 #cell/day
     "nT0_LN": 155.6897 #cell
     "nT1_LN": 113.7387 #cell
     "p0_50": 2.6455e-5 #molecule/micromter2
     "k_M1p0_TCR_off": 1 #1/second
     "phi_M1p0_TCR": 0.09 #1/second
     "k_M1p0_TCR_p":  1 #1/second
     "N_M1p0_TCR": 10 #dimensionless 
     "TCR_p0_tot": 103.813 #molecule/micrometer^2 
     "k_M1p0_TCR_on": 1 # 1/(second*molecule/micrometer^2)  
     
     "Q_nCD4_thym": 7e7 # Thymic output of naive CD4+ T Cells into the blood (cell/day)
     "nCD4_div": 1.16e6 # Naive CD4+ T Cell Diversity (dimensionless)
     "k_nCD4_pro": 3.2e8 # Rate of naive CD4+ T Cell proliferation (cell/day)
     "K_nT_pro": 1e9 # Naive T cell density for half-maximal peripheral proliferation (cell)
     "k_nT_death" = 0.002 # Rate of naive T cell death (1/day)
     
     "q_nCD4_P_out": 5.1 # Naive CD4+ T Cell Transport P->C (1/day)
     "q_nCD4_LN_in": 0.1 # Naive CD4+ T Cell Lymph Node Entry Rate (1/day)
     "q_nCD4_LN_out": 2.88 # Naive CD4+ T Cell Lymph Node Exit Rate (1/day)
     
     "k_nCD4_act": 5 #Maximum Rate of CD4+ Treg Activation by APCs (1/day)
     "IL2_50_Treg": 0.32 #% IL2 Concentration for Half-Maximal Treg Proliferation (nanomolarity)
     "N_IL2_CD4": 8.5 #Additional Number of Activated CD4+ T Cell Generations Due to IL2 (dimensionless)
     "H_CD28_APC": 0.1 #dimensionless

    
    
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
RC["q_nCD4_P_in"] = "k_nT_mig" *"rho_adh" *"gamma_P" * "V_P"
RC["C_total"] = 0.02 * "cell"   



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
        V_T = ((solution[0] + RC["C_total"])*RC["vol_cell"]+(solution[1]+solution[2] + T_total)*RC["vol_Tcell"])/RC["Ve_T"]
        return V_T
    
    

class APC():
'''
    This class contains the number of Antigen-Presneting Cells 
    and the Hill Equations between the APCs and T cells.
'''
    
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
        #Cytokine release in response to the tumour"
        #0 -> c
        r_cytokines_tumor_release = RC["R_Tcell"]*RC["DAMPs"]
        
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

    def Hill_equation(self, solution):
        
        
        #Hill equation between APC and T cell
        H_APC = RC["n_sites_APC"]*self.solution[3]/(RC["n_sites_APC"]*self.solution[3]+RC["nT0_LN"]*RC["n_T0_clones"]+RC["cell"]
        #Hill equation between mAPC and T cell
        H_mAPC = RC["n_sites_APC"]*self.solution[3]/(RC["n_sites_APC"]*self.solution[3]+RC["nT1_LN"]*RC["n_T1_clones"]+RC["cell"]
        #Hill equation between APC and helper T cell
        H_APCh = RC["n_sites_APC"]*self.solution[3]/(RC["n_sites_APC"]*self.solution[3]+RC["nT0_LN"]*RC["n_T1_clones"]+RC["cell"]
                                                     
                                            
        return H_APC, H_mAPC, H_APCh
    
    
class Antigen():
     def __init__(self, initial):
         self.initial = initial
     def antigen_rate(self, t, antigen, V_T, APC_tumor):
     
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
         r_dep = RC["k_dep"]
         #Free antigen degradation
         r_deg_P = RC["k_xP_deg"]*V_T_P*V_T
         #Antigen uptake by mature antigen presenting cells
         r_uptake_T = RC["k_up"]*APC_tumor*V_T_P*V_T 
         r_uptake_e = RC["k_up"]*RC["cell"]*V_T_P*RC["V_e"]
         #Antigen degradation in APC endosomes'
         r_deg_anti = RC["k_P_deg"]*V_e_P*RC["V_e"]
         #Epitope degradation in APC endosomes
         r_deg_epi = RC["k_p_deg"]*V_e_p*RC["V_e"]
         #Antigen-MHC binding in endosome
         r_an_MHC_binding_en = RC["k_on"]*V_e_p*RC["A_e"]*RC["MHC_T"]
         #Antigen-MHC unbinding in endosome
         r_an_MHC_unbinding_en = RC["k_on"]*A_e_Mp*RC["A_e"]
         #Antigen-MHC unbinding on APC surface
         r_an_MHC_unbinding_sur = RC["k_on"]*A_s_Mp*RC["A_s"]
         #Antigen-MHC translocation
         r_an_MHC_trans = RC["kout"]*A_e_Mp*RC["A_e"]
         #MHC translocation
         r_MHC_trans = RC["kout"] * RC["MHC_T"] * RC["A_e"] -RC["kin"] * 1e-6 * RC["A_s"]            
         
         
         #Calculate the ODE to get the antigens values
         dV_e_Pdt = (r_uptake_e - r_deg_anti)/RC["V_e"]
         dV_e_pdt = (r_deg_anti - r_deg_epi - r_an_MHC_binding_en - r_an_MHC_unbinding_en)/RC["V_e"]
         dA_e_Mpdt = (r_an_MHC_binding_en - r_an_MHC_unbinding_en - r_an_MHC_trans )/RC["A_e"]
         dA_s_Mpdt = (r_an_MHC_trans - r_an_MHC_unbinding_sur)/RC["A_s"]
         dA_e_Mdt = (r_an_MHC_unbinding_en - r_an_MHC_binding_en  - r_MHC_trans)/RC["A_e"]
         dA_s_Mdt = (r_an_MHC_unbinding_sur + r_MHC_trans)/RC["A_s"]
         dV_T_Pdt = (r_dep - r_deg_P  - r_uptake_T)/V_T        
         
         return [dV_e_Pdt, dV_e_pdt, dA_e_Mpdt, dA_s_Mpdt, dA_e_Mdt, dA_s_Mdt, dV_T_Pdt]
 
    def simulate(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution = solve_ivp(self.antigen_rate, [t_start, t_end], self.initial, intervals)
        return solution   
   
    
  "k_M1p1_TCR_off" : 1 #1/second
  "phi_M1p1_TCR": 0.09 #1/second
  "k_M1p1_TCR_p": 1 #1/second
  "N_M1p1_TCR": 10 #dimensionless
  "TCR_p1_tot": 103.8131 #molecule/micromiter^2
  "k_M1p1_TCR_on": 1 #1/(second * molecule/micrometer^2)
    
     
     def TCR(self,solution): 
         pTCR_p0_MHC_tot = RC["k_M1p0_TCR_off"]/(RC["k_M1p0_TCR_off"] + RC["phi_M1p0_TCR"]) * (RC["k_M1p0_TCR_p"]/(RC["k_M1p0_TCR_off"]+RC["k_M1p0_TCR_p"]))^RC["N_M1p0_TCR"]*(0.5*(self.solution[3]/RC["n_T0_clones"] + RC["TCR_p0_tot"] + RC["k_M1p0_TCR_off"]/RC["k_M1p0_TCR_on"])/RC["TCR_p0_tot"])^2-4*self.solution[3]/RC["n_T0_clones"]/RC["TCR_p0_tot"])))
         H_Ag = pTCR_p0_MHC_tot/(pTCR_p0_MHC_tot+RC["p0_50"])
         pTCR_p1_MHC_tot = RC["k_M1p1_TCR_off"]/(RC["k_M1p1_TCR_off"] + RC["phi_M1p1_TCR"]) * (RC["k_M1p1_TCR_p"]/(RC["k_M1p1_TCR_off"]+RC["k_M1p1_TCR_p"]))^RC["N_M1p1_TCR"]*(0.5*(self.solution[3]/RC["n_T1_clones"] + RC["TCR_p1_tot"] + RC["k_M1p1_TCR_off"]/RC["k_M1p1_TCR_on"])/RC["TCR_p1_tot"])^2-4*self.solution[3]/RC["n_T1_clones"]/RC["TCR_p1_tot"])))
         H_Agh = pTCR_p1_MHC_tot/(pTCR_p1_MHC_tot+RC["p1_50"])
         
         return H_Ag, H_Agh
     
        
         
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


class Treg():
    def __init__(self, initials):
        self.initial = initial

    def naive_Treg(self, nTreg, H_APC):
        
        V_P_nTreg = nTreg[0]
        V_LN_nTreg = nTreg[1]
        V_C_nTreg = nTreg[2]

        # Those are the reactions for T cell.
        # Thymic output of naive T cell to blood
        r_nTreg_thymic= RC["Q_nCD4_thym"]/RC["nCD4_div"]
        # Naive T cell proliferation in the peripheral compartment
        r_nTreg_pro_pheripheral= RC["k_nCD4_pro"]/RC["nCD4_div"] * V_P_nTreg/(RC["K_nT_pro"]/RC["nCD4_div"] + V_P_nTreg)
        # Naive T cell proliferation in the TDLN compartment
        r_nTreg_pro_TDLN = RC["k_nCD4_pro"]/RC["nCD4_div"]* V_LN_nTreg/(RC["K_nT_pro"]/RC["nCD4_div"] + V_LN_nTreg)
        # Naive T cell death in the peripheral compartment
        r_nTreg_death_pheripheral = RC["k_nT_death"] * V_P_nTreg
        # Naive T cell death in the central compartment
        r_nTreg_death_central = RC["k_nT_death"] * V_C_nTreg
        # Naive T cell death in the TDLN compartment
        r_nTreg_death_TDLN = RC["k_nT_death"] * V_LN_nTreg
        # Naive T cell entry into the peripheral compartment
        r_nTreg_entry_peripheral = RC["q_nCD4_P_in"] * V_C_nTreg
        # Naive T cell exit from the peripheral compartment
        r_nTreg_exit_peripheral = RC["q_nCD4_P_out"] * V_P_nTreg
        # Naive T cell transport into the lymph node
        r_nTreg_entry_TDLN = RC["q_nCD4_LN_in"] * V_C_nTreg
        # Naive T cell exit from the lymph node
        r_nTreg_exit_TDLN = RC["q_nCD4_LN_out"] * V_LN_nTreg
        #Naive T cell activation
        r_nT_APC_antigen = RC["k_nCD4_act"] * H_APC * H_Ag * V_LN_nTreg
        
        #Calculated the ODEs for the naive Treg cells.
        dV_C_nTregdt = r_nTreg_thymic - r_nTreg_entry_peripheral + r_nTreg_exit_peripheral - r_nTreg_entry_TDLN + r_nTreg_exit_TDLN - r_nTreg_death_central
        dV_P_nTregdt = r_nTreg_pro_pheripheral + r_nTreg_entry_peripheral - r_nTreg_exit_peripheral - r_nTreg_death_pheripheral
        dV_LN_nTregdt = r_nTreg_pro_TDLN + r_nTreg_entry_peripheral - r_nTreg_exit_peripheral - r_nTreg_death_TDLN - r_nTreg_APC_antigen
        return [dV_C_nTregdt, dV_P_nTregdt, dV_LN_nTregdt]
    
    def simulate_nTreg(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution_nT = solve_ivp(self.naive_Treg, [t_start, t_end], self.initial, intervals)
        return solution_nTreg   
    
    def Treg_activation(self, Treg, H_APC, H_Ag, V_LN_nTreg):
        
        
        V_LN_aTreg = Treg[0]
        V_LN_Treg = Treg[1]
        V_C_Treg = Treg[2]
        V_P_Treg = Treg[3]
        V_T_Treg = Treg[4]
        V_LN_IL2_reg = Treg[5]
      
        
        #Those reactions are for T cells to be activated. 
        #Naive T cell activation
        r_nT_act = RC["k_nCD4_act"]* H_APC * H_Ag *V_LN_nTreg * RC["n_clones_slf"]
        #Activated T cells proliferation 
        r_aT_pro = RC["k_CD4_pro"]/N_aT0*V_LN_aTreg
        r_aT_pro_ad = (r_aT_pro) * 2^(N_aT0)
        #T cell death in the central compartment
        r_T_death_central = RC["k_CD4_death"] * V_C_Treg
        #T cell death in the lymph node compartment
        r_T_death_LN = RC["k_CD4_death"] * V_LN_Treg
        #T cell death in the tumor compartment
        r_T_death_T = RC["k_CD4_death"] * V_T_Treg
        #T cell death in the pherical compartment
        r_T_death_P = RC["k_CD4_death"] * V_P_Treg
        #T cell clearance upon antigen clearance
        r_T_clear_Ag = RC["k_cell_clear"] * V_T_Treg * (RC["Kc_rec"]/(RC["C_total"]^2 + RC["Kc_rec"]))
        #T cell transport into the peripheral compartment
        r_T_in_pher = RC["q_CD4_P_in"] * V_C_Treg
        #T cell transport out of the peripheral compartment
        r_T_out_pher = RC["q_CD4_P_out"]*V_P_Treg
        #T cell transport into the tumor compartment
        r_T_in_tumor = RC["q_CD4_T_in"] * V_C_Treg * (RC["C_total"]^2/(RC["C_total"]^2 + RC["Kc_rec"]))
        #T cell transport out of the lymph node compartment
        r_T_out_LN = RC["q_CD4_LN_out"]*V_LN_Treg
        #IL2 degradation
        r_IL2_degra = RC["k_IL2_deg"]*V_LN_IL2_reg*RC["V_LN"]
        #IL2 consumption by Treg cells
        r_IL2_com = RC["k_IL2_cons"]*V_LN_Treg*V_LN_IL2_reg/(RC["IL2_50_Treg"]+V_LN_IL2_reg)                               
        #IL2 secretion from activated T cells
        r_IL12_sec = RC["k_IL2_sec"]*V_LN_aTreg
        
        
        
        #Those are calclated to solve the parameters in the list by ODE.
        dV_LN_aTregdt = r_nT_act - r_T_out_LN
        dV_LN_Tregdt = r_aT_pro_ad - r_T_out_LN - r_T_death_LN
        dV_C_Tregdt = r_T_out_LN - r_T_in_pher + r_T_out_pher - r_T_in_tumor - r_T_death_central
        dV_P_Tregdt = r_T_in_pher - r_T_out_pher - r_T_death_P 
        dV_T_Tregdt = r_T_in_tumor - r_T_death_T - r_T_clear_Ag
        dV_LN_IL2_regdt = r_IL12_sec - r_IL2_degra - r_IL2_com
        
        return [dV_LN_aTregdt, dV_LN_Tregdt, dV_C_Tregdt,dV_P_Tregdt,  dV_T_Tregdt, dV_LN_IL2_regdt]
    

    def simulate_aT(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution_aT = solve_ivp(self.Treg_activation, [t_start, t_end], self.initial, intervals)
        N_aT = RC["N0"] + RC["N_costim"]* RC["H_CD28_APC"] + RC["N_IL2_CD4"]*self.solution[5]/(RC["IL2_50"]+self.solution[5])
        return solution_aT   
    
    
class Thelper():
    def __init__(self, initials):
        self.initial = initial

    def active_Th(self, nTh, H_APCh, V_LN_Treg, H_Agh):


        # Those are the parameters needed.
        "k_Th_act": 10 # T helper cell activation rate (1/day)
        k_T0_pro =
        k_Th_Treg = 0.022 # Th differentiation rate to Treg (1/day)
        k_cell_clear = 0.1 # [T cell clearance upon Ag clearance] Dead Cell Clearance Rate (1/day)
        q_T0_P_in =
        q_T0_P_out =
        Kc_rec = 2.02e7 # Half-Maximal cancer cell number for T cell recruitment (cell^2)
        q_T0_T_in =
        q_T0_LN_out =



        # Those are the reactions for T cell activation.
        # Naive T cell activation
        r_nT_act = RC["k_Th_act"] * H_APCh * H_Agh * V_LN_Treg * RC["n_T1_clones"]
        # Activated T cell proliferation
        = (k_T0_pro/N_aTh) * V_LN.aT
        = (k_T0_pro/N_aTh) * 2^N_aTh * V_LN.aT
        # T cell clearance upon Ag(antigen) clearance
        = k_cell_clear * V_T.T * (Kc_rec/(C_total^2 + Kc_rec))
        # T cell transport into the peripheral compartment
        = q_T0_P_in * V_C.T
        # T cell transport out of the peripheral compartment
        = q_T0_P_out * V_P.T
        # T cell transport into the tumor compartment
        = q_T0_T_in * V_T * V_C.T * (C_total^2 / (C_total^2 + Kc_rec))
        # T cell transport out of the lymph node compartment
        = q_T0_LN_out * V_LN.T


              
         
class Teff():
    def __init__(self, initials):
        self.initial = initial
        
    def naive_T(self, nT, H_APC, H_Ag): 
        
        #Those calculated parameters are saved in the matrix.
        V_C_nT = nT[0]
        V_P_nT = nT[1]
        V_LN_nT = nT[2]
        
        
        #Those are the reactions for T cell.
        #Those part is for naive T cell dynamics
        #Thymic output of naive T cell to blood
        r_nT_thymic = RC["Q_nCD8_thym"]/RC["nCD8_div"]
        #Naive T cell proliferation in the peripheral compartment
        r_nT_pro_pheripheral = RC["k_nCD8_pro"]/RC["nCD8_div"]*V_P_nT/(RC["K_nT_pro"]/RC["nCD8_div"]+V_P_nT
        #Naive T cell proliferation in the TDLN compartment
        r_nT_pro_TDLN = RC["k_nCD8_pro"]/RC["nCD8_div"]*V_LN_nT/(RC["K_nT_pro"]/RC["nCD8_div"]+V_LN_nT)                                        
        #Naive T cell death in the peripheral compartment
        r_nT_death_pheripheral = RC["k_nT_death"]*V_P_nT
        #Naive T cell death in the central compartment
        r_nT_death_central = RC["k_nT_death"]*V_C_nT
        #Naive T cell death in the TDLN compartment
        r_nT_death_TDLN = RC["k_nT_death"]*V_LN_nT
        #Naive T cell entry into the peripheral compartment
        r_nT_entry_peripheral = RC["q_nCD8_P_in"]*V_C_nT
        #Naive T cell exit from the peripheral compartment
        r_nT_exit_peripheral = RC["q_nCD8_P_out"]*V_P_nT
        #Naive T cell entry into the lymph node
        r_nT_entry_TDLN = RC["q_nCD8_LN_in"]*V_C_nT
        #Naive T cell exit from the lymph node
        r_nT_exit_TDLN = RC["q_nCD8_LN_out"]*V_LN_nT
        #Naive T cell activation
        r_nT_APC_antigen = RC["k_nCD8_act"]* H_APC *H_Ag *V_LN_nT
        
        
        
        #Calculated the ODEs for the naive Tcells.
        dV_C_nTdt = r_nT_thymic - r_nT_entry_peripheral + r_nT_exit_peripheral - r_nT_entry_TDLN + r_nT_exit_TDLN - r_nT_death_central
        dV_P_nTdt = r_nT_pro_pheripheral + r_nT_entry_peripheral - r_nT_exit_peripheral - r_nT_death_pheripheral
        dV_LN_nTdt = r_nT_pro_TDLN + r_nT_entry_peripheral - r_nT_exit_peripheral - r_nT_death_TDLN - r_nT_APC_antigen
        return [dV_C_nTdt, dV_P_nTdt, dV_LN_nTdt]
    
    def simulate_nT(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution_nT = solve_ivp(self.naive_T, [t_start, t_end], self.initial, intervals)
        return solution_nT   
    
   
    
    def T_activation(self, Tact, H_APC, H_Ag, V_LN_nT,H_PD1_C1, Tregs):
        
        
        V_LN_aT = Tact[0]
        V_LN_T = Tact[1]
        V_C_T = Tact[2]
        V_P_T = Tact[3]
        V_T_T = Tact[4]
        V_LN_IL2 = Tact[5]
      
        
        #Those reactions are for T cells to be activated. 
        #Naive T cell activation
        r_nT_act = RC["k_nCD8_act"]* H_APC * H_Ag *V_LN_nT * RC["n_clones_tum"]
        #Activated T cells proliferation 
        r_aT_pro = RC["k_CD8_pro"]/N_aT*V_LN_aT
        r_aT_pro_ad = (r_aT_pro) * 2^(N_aT)
        #T cell death in the central compartment
        r_T_death_central = RC["k_CD8_death"] * V_C_T
        #T cell death in the lymph node compartment
        r_T_death_LN = RC["k_CD8_death"] * V_LN_T
        #T cell death in the tumor compartment
        r_T_death_T = RC["k_CD8_death"] * V_T_T
        #T cell death in the pherical compartment
        r_T_death_P = RC["k_CD8_death"] * V_P_T
        #T cell clearance upon antigen clearance
        r_T_clear_Ag = RC["k_cell_clear"] * V_T_T * (RC["Kc_rec"]/(RC["C_total"]^2 + RC["Kc_rec"]))
        #T cell death from Tregs 
        r_T_death_Treg = RC["k_Treg"] * V_T_T * Tregs/(V_T_T+Tregs+RC["cell"])
        #T cell death from cancer
        r_T_death_cancer = RC["k_Tcell"]*V_T_T*RC["C_total"]/(RC["C_total"]+V_T_T+RC["cell"])*H_PD1_C1
        #T cell transport into the peripheral compartment
        r_T_in_pher = RC["q_CD8_P_in"] * V_C_T
        #T cell transport out of the peripheral compartment
        r_T_out_pher = RC["q_CD8_P_out"]*V_P_T
        #T cell transport into the tumor compartment
        r_T_in_tumor = RC["q_CD8_T_in"] * V_C_T * (RC["C_total"]^2/(RC["C_total"]^2 + RC["Kc_rec"]))
        #T cell transport out of the lymph node compartment
        r_T_out_LN = RC["q_CD8_LN_out"]*V_LN_T
        #IL2 degradation
        r_IL12_sec = RC["k_IL2_deg"]*V_LN_IL2*RC["V_LN"]
        #IL2 consumption by T cells
        r_IL2_degra = RC["k_IL2_cons"]*V_LN_T*V_LN_IL2/(RC["IL2_50"]+V_LN_IL2
        #IL2 secretion from activated T cells
        r_IL2_com = RC["k_IL2_sec"]*V_LN_aT
        
        
        
        #Those are calclated to solve the parameters in the list by ODE.
        dV_LN_aTdt = r_nT_act - r_T_out_LN
        dV_LN_Tdt = r_aT_pro_ad - r_T_out_LN - r_T_death_LN
        dV_C_Tdt = r_T_out_LN - r_T_in_pher + r_T_out_pher - r_T_in_tumor - r_T_death_central
        dV_P_Tdt = r_T_in_pher - r_T_out_pher - r_T_death_P 
        dV_T_Tdt = r_T_in_tumor - r_T_death_T - r_T_death_Treg - r_T_death_cancer - r_T_clear_Ag
        dV_LN_IL2dt = r_IL12_sec - r_IL2_degra - r_IL2_com
        
        return [dV_LN_aTdt, dV_LN_Tdt, dV_C_Tdt, dV_P_Tdt, dV_T_Tdt, dV_LN_IL2dt]
    

    def simulate_aT(self,t_start = 0, t_end = 400): #unit for time is day
        intervals = np.linspace(t_start, t_end, 401)
        solution_aT = solve_ivp(self.nT_activation, [t_start, t_end], self.initial, intervals)
        N_aT0 = RC["N0"] + RC["N_costim"]* RC["H_CD28_APC"] + RC["N_IL2_CD8"]*self.solution[5]/(RC["IL2_50"]+self.solution[5])
        return solution_aT   



  

  
    
   
   
