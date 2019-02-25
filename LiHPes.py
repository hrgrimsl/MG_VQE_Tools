import os
from sys import argv

script, filename = argv

for i in range(5, 25):
    r = .2*i
    string = filename+'_'+str(i)+'_1_PT2'
    write = open(string, 'w')
    write.write('basis sto-3g \n')
    write.write('multiplicity 1 \n')
    write.write('charge 0 \n')
    write.write('psi_file '+string+' \n')
    geo_string = ''
    geo_string+= 'atom Li 0 0 0\n'
    geo_string+= 'atom H 0 0 '+str(r)
    write.write(geo_string+' \n')
    write.write('op_kwarg include_pqrs True \n')
    write.write('op_kwarg screen_commutators False \n')
    write.write('op_kwarg spin_adapt True \n')
    write.write('method_kwarg algorithm ADAPT_End \n')
    write.write('method_kwarg theta_tightness 1e-7 \n')
    write.write('method_kwarg ADAPT_tightness 1e-1 \n')
    write.write('method_kwarg RADAPT_Seed 0\n')
    write.close()

        
for i in range(5, 25):
    r = .2*i
    string = filename+'_'+str(i)+'_2_PT2'
    write = open(string, 'w')
    write.write('basis sto-3g \n')
    write.write('multiplicity 1 \n')
    write.write('charge 0 \n')
    write.write('psi_file '+string+' \n')
    geo_string = ''
    geo_string+= 'atom Li 0 0 0\n'
    geo_string+= 'atom H 0 0 '+str(r)
    write.write(geo_string+' \n')
    write.write('op_kwarg include_pqrs True \n')
    write.write('op_kwarg screen_commutators False \n')
    write.write('op_kwarg spin_adapt True \n')
    write.write('method_kwarg algorithm ADAPT_PT2 \n')
    write.write('method_kwarg theta_tightness 1e-7 \n')
    write.write('method_kwarg ADAPT_tightness 1e-2 \n')
    write.write('method_kwarg RADAPT_Seed 0\n')
    write.close()

        
    
        
    
        
    
