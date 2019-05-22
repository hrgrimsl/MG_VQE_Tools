import os
from sys import argv

script, filename = argv

for i in range(5, 21):
    r = .1*i
    string = filename+'_'+str(i)
    write = open(string, 'w')
    write.write('basis sto-3g \n')
    write.write('multiplicity 1 \n')
    write.write('charge 0 \n')
    write.write('psi_file B2scratch'+str(i)+'\n')
    geo_string = ''
    geo_string+= 'atom H 0 0 0\n'
    geo_string+= 'atom H 0 0 '+str(1*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(2*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(3*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(4*r)+'\n'
    geo_string+= 'atom H 0 0 '+str(5*r)+'\n'
    write.write(geo_string+' \n')
    write.write('op_kwarg include_pqrs True \n')
    write.write('op_kwarg screen_commutators False \n')
    write.write('op_kwarg spin_adapt True \n')
    write.write('op_kwarg active 2,3,4,5,6,7,8,9 \n')
    write.write('op_kwarg occs 2,3,4 \n')
    write.write('op_kwarg spin_adapt True \n')
    write.write('method_kwarg algorithm CASCI \n')
    write.write('method_kwarg theta_tightness 1e-7 \n')
    write.write('method_kwarg ADAPT_tightness 1e-3 \n')
    write.write('method_kwarg RADAPT_Seed 0\n')
    write.write('Singular_threshold 1e-10')
    write.close()
    os.system('./Driver.py '+str(string)+' '+str(string)+'.out \n') 
        
    
        
    
