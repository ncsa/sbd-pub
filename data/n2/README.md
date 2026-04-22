This directory has input data for the N2 molecule using the 6-31g basis set.  The fcidump.txt file was obtained from a restricted Hartree-Fock calculation using pyscf.  A full CI (FCI) calculation using pyscf reported electronic energy E = -109.048742 Ha.  Alpha bitstrings were selected by filtering the results of the FCI calculation, saving only the configurations that had coefficients with absolute value greater than some threshold.  The table below lists the threshold value, number of alpha bitstrings, the total number of determinants, and the electronic energy.


| threshold|A bitstrings| Determinants| Energy(Ha)  |
|----------|------------|-------------|-------------|
|   1.0e-3 |     239    |    5.71e4   |-109.04162110|
|   3.0e-4 |     642    |    4.12e5   |-109.04697304|
|   1.0e-4 |    1387    |    1.92e6   |-109.04835269|
|   3.0e-5 |    2583    |    6.67e6   |-109.04864315|
|   1.0e-5 |    4041    |    1.63e7   |-109.04871934|
|   3.0e-6 |    6930    |    4.80e7   |-109.04873940|
|   1.0e-6 |    9332    |    8.71e7   |-109.04874139|
|   3.0e-7 |   13454    |    1.81e8   |-109.04874191|
|   1.0e-7 |   17558    |    3.08e8   |-109.04874198|
|    FCI   |   31824    |    1.01e9   |-109.04874199|


The computational work increases faster than linear in the number of determinants because the number of non-zero entries in the Hamiltonian matrix increases somewhat as the number of bitstrings increases.  Our measurements suggest an approximate scaling of compute time ~(determinants)^1.23.  The bitstring files corresponding to the table entries are included : threshold = 1.0e-4  => 1em4-alpha.txt, etc.  The input file 1em4-alpha.txt has 1387 alpha bitstrings resulting in 1.92e6 determinants, yielding an energy ~0.4 mHa higher than the pyscf FCI result.  
