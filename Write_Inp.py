import os
from sys import argv

filename, script = argv

for i in range(5, 25):
    r = .2*i
    string = filename+'_'+str(i)
    write = open(string, 'w')
    write.write('basis sto-3g')
    write.write('multiplicity = 1')
    write.write('psi_file = '+string)
    geo_string = ''
    geo_string+= 'Li 0 0 0\n'
    geo_string+= 'H 0 0 '+str(r)
    write.write(geo_string)
    write.write('op_kwarg include_pqrs True')
    write.write('op_kwarg screen_commutators False')
    write.write('op_kwarg spin_adapt True')
    write.write('method_kwarg algorithm ADAPT')
    write.write('method_kwarg theta_tightness 1e-7')
    write.write('method_kwarg ADAPT_tightness 1e-3')
    write.write('method_kwarg RADAPT_Seed')
    
    r = .2*i
    string = filename+'_'+str(i)+'_UCCSD'
    write = open(string, 'w')
    write.write('basis sto-3g')
    write.write('multiplicity = 1')
    write.write('psi_file = '+string)
    geo_string = ''
    geo_string+= 'Li 0 0 0\n'
    geo_string+= 'H 0 0 '+str(r)
    write.write(geo_string)
    write.write('op_kwarg include_pqrs True')
    write.write('op_kwarg screen_commutators False')
    write.write('op_kwarg spin_adapt True')
    write.write('method_kwarg algorithm UCC')
    write.write('method_kwarg theta_tightness 1e-7')
    write.write('method_kwarg ADAPT_tightness 1e-3')
    write.write('method_kwarg RADAPT_Seed')
    
        
    
        
    
