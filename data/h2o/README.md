This directory has input data for the H2O molecule using the cc-pvdz basis set.  The fcidump.txt file was obtained from a restricted Hartree-Fock calculation using pyscf.  A full CI (FCI) calculation using pyscf reported electronic energy E = -76.2437768 Ha.  Alpha bitstrings were selected by filtering the results of the FCI calculation, saving only the configurations that had coefficients with absolute value greater than some threshold.  The table below lists the threshold value, number of alpha bitstrings, the total number of determinants, and the electronic energy.  There are 24 orbitals with 5 alpha plus 5 beta electrons, so the number of determinants for the full CI calculation is binomial(24,5)^2 = 1.81e9.  For the Davidson algorithm with maximum subspace dimension set to 10, one needs about 25 vectors with dimension = number of determinants; so the memory required for 1.52e9 determinants would be ~ 25 * 1.52e9 * 8 bytes => ~300 GB.


| threshold|A bitstrings| Determinants| Energy(Ha) |
|----------|------------|-------------|------------|
|   1.0e-3 |     275    |    7.56e4   | -76.2359376|
|   1.0e-4 |    1543    |    2.38e6   | -76.2429495|
|   1.0e-5 |    5514    |    3.04e7   | -76.2437261|
|   1.0e-6 |   13822    |    1.91e8   | -76.2437670|
|   1.0e-7 |   25062    |    6.28e8   | -76.2437678|
|   1.0e-8 |   39028    |    1.52e9   |   t.b.d.   |
|    FCI   |   42504    |    1.81e9   | -76.2437768|


The computational work increases faster than linear in the number of determinants because the number of non-zero entries in each row of the Hamiltonian matrix increases somewhat as the number of bitstrings increases.  Our measurements suggest an approximate scaling of compute time of ~(determinants)^1.23.  The bitstring files corresponding to the table entries are included : threshold = 1.0e-4  => h2o-1em4-alpha.txt, etc.  The input file h2o-1em4-alpha.txt has 1543 alpha bitstrings resulting in 2.38e6 determinants, yielding an energy ~0.8 mHa higher than the pyscf FCI result.  This input file is a good choice for single-node tests.