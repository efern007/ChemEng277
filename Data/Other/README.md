This dataset contains raw data and capacity extractions for all 92 cells presented in the paper "Dynamic cycling enhances battery lifetime". 
If you use this dataset, please cite the paper: Geslin, A., Xu, L., Ganapathi, D. et al. Dynamic cycling enhances battery lifetime. Nat Energy (2024). https://doi.org/10.1038/s41560-024-01675-8

The file diagnostic_capacities.pkl is a multi_index dataframe containing the capacities for all cells for all diagnostic cycles (both CC charge and CC discharge capacities at C/40 and C/2, during the diagnostic cycles). All capacities are normalized by the cell nominal capacity. The C-rates and capacities can thus be scaled with the nominal capacity of your choice. 

The protocol_mapping_dic.json can be used to map the cell number to the type of protocol run on that cell. 

For each cell, there are two files.
1: the raw data file contains the raw, unprocessed data. Time, Normalized current (C-rate), Voltage, and normalized capacity are reported."Cyc#", "Step", "State", "Full Step #", "Loop3" can be used to identify subparts of a cycle for each cycle. In particular, "Loop3" can be used to cycle through repeats of aging protocol elementary units used during discharge to reach the lower voltage limit. Additionally, while aging cycles are intuitively assigned to only one cycle number (Cyc#), it is worth mentioning that two cycles numbers (Cyc#) are associated with each diagnostic routine. The first of these cycle numbers is used to refer to the corresponding row in the diagnostic_capacities.pkl file.  
2: An aging summary, which contains the charging and discharging capacity for all cells, along with the cumulative capacity throughput for each cycle. All capacities are normalized as described earlier. NB: the discharging capacity is the total capacity passed through during discharge (which includes the regenerative portions). Routine diagnostic cycles contains multiple charges-discharges, but are only reported over two "cycles numbers" (#Cyc) which will have higher capacities than the rest of the aging "Cycles", in particular, the initial diagnostic cycles (cycles 1 and 2).





 