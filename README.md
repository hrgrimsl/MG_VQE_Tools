Mayhall & Grimsley VQE Tools
===
Structuring An Input File
===
basis
---
Define your basis set.  Works with any basis set that OFPsi4 can handle.

multiciplity
---
Define multiplicity.

charge
---
Define charge.

psi\_file
---
Define scratch file for the OFPsi4 calculation- be careful not to run two calculations with the same scratch file at the same time.  Additionally, MAGE will recycle calculations from before if there is already a psi4 file of the right name.

atom
---
e.g. 'atom H 0 0 0'
Adds an atom with the Cartesians (angstroms) specified.

op\_kwargs:
---
include\_pqrs - Use GSD?
spin\_adapt - Spin-adapt operators?
active - Specifies spatial orbitals in active space.  Should be a comma-separated string.  Needs to be used in conjunction with occs and soccs.  Any leftover electrons are assumed to fill the bottom n molecular orbitals.
occs - Doubly occupied spatial orbitals in active space.  False if none.
soccs - Singly occupied spatial orbitals in active space.  False if none.

  
method\_kwargs:
---
algorithm - What algorithm to use?   VQE, UCC, or ADAPT?
theta\_tightness- Convergence requirement for thetas.
ADAPT\_tightness- Convergence requirement for adding terms to ADAPT ansatz.






Contact Us
===
hrgrimsl@vt.edu

