import hashlib as hsh

def show_matrix_hash(self,*matr)->str:
    hash_obj=hsh.sha256()
    matr_list=['']*len(matr)
    j=0
    #:matrix<R> 
    for i  \
        in matr:   
             hash_obj.update(str(matr).encode('ascii'))
             matr_list[j]=hash_obj.hexdigest()
             j+=1
    return matr_list        
