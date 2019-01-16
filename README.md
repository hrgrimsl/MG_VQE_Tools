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
Define scratch file for the OFPsi4 calculation- be careful not to run two calculations with the same scratch file at the same time.

atom
---
e.g. 'atom H 0 0 0'
Adds an atom with the Cartesians (angstroms) specified.

op\_kwargs:
---
include\_pqrs - Use GSD?
screen\_commutators - Remove small commutators?
spin\_adapt - Spin-adapt operators? (Only supports singlets for now.)

method\_kwargs:
---
algorithm - What algorithm to use?   VQE, UCC, or ADAPT?
theta\_tightness- Convergence requirement for thetas.
ADAPT\_tightness- Convergence requirement for adding terms to ADAPT ansatz.
RADAPT\_seed- If using RADAPT, what seed to use?





Contact Us
===
hrgrimsl@vt.edu

