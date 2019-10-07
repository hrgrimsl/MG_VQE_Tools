import os
from sys import argv

script, filename, seed = argv

for i in range(0, 60):
    r = .5+.1*i
    string = filename+'_'+str(i)
    write = open(string, 'w')
    write.write('basis sto-3g \n')
    write.write('multiplicity 1 \n')
    write.write('charge 0 \n')
    write.write('output '+string+str(r)+'\n')
    write.write('psi_file '+string+str(r)+'\n')
    geo_string = ''
    geo_string+= 'atom H 0 0 0\n'
    geo_string+= 'atom H 0 0 '+str(r)+'\n'
    geo_string+= 'atom H 0 0 '+str(2*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(3*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(4*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(5*r)+'\n'
    write.write(geo_string)
    write.write('op_kwarg include_pqrs False \n')
    write.write('op_kwarg spin_adapt True \n')
    write.write('op_kwarg theta_tightness 1e-7\n')
    write.write('op_kwarg ADAPT_tightness 1e-30\n')
    write.write('method_kwarg algorithm UCC\n')
    write.write('op_kwarg repeats False\n')
    write.write('method_kwarg theta_tightness 1e-7 \n')
    write.write('method_kwarg ADAPT_tightness 1e-30 \n')
    write.write('n_fdoccs 0\n')
    write.write('op_kwarg inter '+str(seed))
    write.close()

        
    
        
    
