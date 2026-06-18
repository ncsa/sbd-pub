This directory has inputs for the C4H4 molecule (singlet ground state in a rectangular geometry) using the 6-31g basis set.  There are 44 orbitals and 28 electrons, so the dimension of the full CI space, binomial(44,14) squared = 6.70e18, is too large for direct solution.  The TrimCI code was used to generate bit-strings. We use the unique list of bit-strings obtained by merging the lists of alpha and beta bit-strings from the core set of determinants produced by TrimCI.  The tensor product is formed to produce the sub-spaced used for Davidson diagonalization.  The results are summarized in the table below, where "TP" is for tensor-product.  For comparison, PySCF yielded Hartree-Fock energy  -153.57176996 H, and CCSD(T) energy = -153.99141235 H.


|TCI dets|  TCI Energy | alpha bit-strings| TP dimension|  TP Energy  |
|--------|-------------|------------------|-------------|-------------|
|  30000 |-153.91341920|      5226        |   2.73e7    |-153.95253343|
| 100000 |-153.93507332|     14690        |   2.16e8    |-153.96629869|
