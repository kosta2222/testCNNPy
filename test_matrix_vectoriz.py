import numpy as np
import hash_matr as hm

def sigmoid(val:float)->float:

        return (1.0/(1.0+np.exp(-val)))
def makeHidden(signals:np.ndarray,weights:np.ndarray)->(np.ndarray,np.ndarray):
    cost=np.dot(weights,signals)

    cost_activ=np.zeros((cost.shape[0],cost.shape[1]))

    i=0
    for row in cost:
            for elem in row:
              cost_activ[i,0]=sigmoid(elem)
              i+=1        

    return (cost,cost_activ)  

weights1=np.random.normal(0.0,2**-0.5,size=(2,5))
signals1=np.random.normal(0.0,2**-0.5,size=(5,1))

weights2=np.random.normal(0.0,2**-0.5,size=(2,5))
signals2=np.random.normal(0.0,2**-0.5,size=(5,1))

like1=makeHidden(signals1,weights1)
print(like1)
print(hm.show_matrix_hash(like1[1]))
like2=makeHidden(signals2,weights2)
print(like2)
print(hm.show_matrix_hash(like2[1]))
