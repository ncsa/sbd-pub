This directory has inputs for the C2H2 molecule (singlet ground state in a linear geometry) using the cc-pvdz basis set.  There are 38 orbitals and 14 electrons, so the dimension of the full CI space, binomial(38,7) squared = 1.59e14, is too large for direct solution.  The TrimCI code was used to generate bit-strings. We use the unique list of bit-strings obtained by merging the lists of alpha and beta bit-strings from the core set of determinants produced by TrimCI.  The tensor product is formed to produce the sub-spaced used for Davidson diagonalization.  The results are summarized in the table below, where "TP" is for tensor-product.  For comparison, PySCF yielded Hartree-Fock energy -76.823797097 H, and CCSD(T) energy = -77.115206898 H.


|TCI dets| TCI Energy | alpha bit-strings| TP dimension| TP Energy  |
|--------|------------|------------------|-------------|------------|
|  30000 |-77.09356184|      4116        |   1.69e7    |-77.10918904|
| 100000 |-77.10188269|      7868        |   6.19e7    |-77.11337996|
