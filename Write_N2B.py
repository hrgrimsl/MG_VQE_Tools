import os
from sys import argv

script, filename, seed = argv

for i in range(0, 30):
    r = .5+.15*i
    string = filename+'_'+str(i)
    write = open(string, 'w')
    write.write('basis sto-3g \n')
    write.write('multiplicity 1 \n')
    write.write('charge 0 \n')
    write.write('output '+string+str(r)+'\n')
    write.write('psi_file '+string+str(r)+'.psi\n')
    geo_string = ''
    geo_string+= 'atom N 0 0 0\n'
    geo_string+= 'atom N 0 0 '+str(r)+'\n'

    write.write(geo_string)
    write.write('op_kwarg include_pqrs False \n')
    write.write('op_kwarg spin_adapt True \n')
    write.write('method_kwarg algorithm VQE\n')
    if r<.9:
        write.write('op_kwarg active 4,5,6,7,8,9 \n') 
        write.write('op_kwarg reorder 4,3,6,7,8,9 \n') 
    else:
        write.write('op_kwarg active 4,5,6,7,8,9 \n') 
        write.write('op_kwarg reorder 4,5,6,7,8,9 \n') 
    write.write('method_kwarg theta_tightness 1e-7 \n')
    write.write('method_kwarg ADAPT_tightness 1e-5 \n')
    write.write('n_fdoccs 4\n')
    write.write('op_kwarg intra '+str(seed)+'\n')
    #write.write('op_kwarg flip True\n')
    write.close()

        
    
        
    
