# Final-Project
Immune System is a complicated system, and it plays a really important role when we take various medicine. 
So, basically, we decide to stimulate how the immune system works after taking atezolizumab and nab-paclitaxel. 
By comparing the tumor size, amount of T cells and APCs in tumor compartment, we could know which medicine works better. 
Therefore, four module is established to mimic the immune system, which are the cancer module, 
the T cell module, and the APC module, and the checkpoints module. Each module are connected and 
placed in one function. Different parts are connected by Hill Equations. By adding atezolizumab, checkpoint affinity will get influenced
and by adding nab-paclitaxel, a chemotherapy medicine, is working on changing the cytokines producing rate. By running the code, we can get three set of data of 'Cancer Size', 'No. of T effective in tumor compartment', and 'No. of Antigen presenting cell(APC) in tumor compartment'.

<img width="416" alt="image" src="https://user-images.githubusercontent.com/113488305/208579336-ab3b3703-06dc-4bfc-9fc3-db920dbb22f5.png">

Two modules will be done by each person and connection between various elements in the immune system is listed below. 
(Written by Matlab)
All the data is from the Github link listed below and the paper we mimiced is [1].
'''https://github.com/popellab/QSPIO-TNBC/tree/main/parameters'''

Reference:
[1]Wang H, Ma H, Sové RJ, et al. Quantitative systems pharmacology model predictions for efficacy of atezolizumab and nab-paclitaxel in triple-negative breast cancer. Journal for ImmunoTherapy of Cancer 2021;9:e002100. doi: 10.1136/jitc-2020-002100
