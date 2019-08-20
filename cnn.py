import numpy as np
import traceback as t


F=3
F1=2
W=7
    
d=3
S=1
P=0
    
b0=+1
    
np.random.seed(42)
    
X=np.random.randint(100,size=(W,W,d))
    
W0=np.random.randint(100,size=(F,F,d))

W1=np.random.randint(100,size=(F1,F1,d))
    
def relu(val:float):
        if val<0:
            return 0
        else:
            return val
    
def derivate_relu(val:float):
        if val<0:
                return 0.001
        else:
                return 1.0
        
        
    
def Output1(X:np.ndarray,W0:np.ndarray)->np.ndarray:
       W=X.shape[0]
       F=W0.shape[0]
       P=0
       
       OutV=int((W-F+2*P)/S+1)
       OutDepth=1
       
       V1=np.zeros((OutV,OutV,OutDepth))
       
       for downY in range(OutV):
           for acrossX in range(OutV):
               V1[downY,acrossX,0]=np.sum(X[S*downY:S*downY+F,S*acrossX:S*acrossX+F,:]*W0)+b0
               
       return V1
   
def Output2(V1:np.ndarray)->np.ndarray:
       V2=np.zeros((V1.shape[0],V1.shape[0],V1.shape[2]))
       
       for row in range(V1.shape[0]):
           for elem in range(V1.shape[1]):
               V2[row][elem][0]=relu(V1[row][elem][0])
       return V2
   
def Maxpooling(X1:np.ndarray,W1:np.ndarray,S)->np.ndarray:
       W=X1.shape[0]
       F1=W1.shape[0]
       P=0
       OutV=int((W-F1+2*P)/S+1)
       OutDepth=1

       V3=np.zeros((OutV,OutV,OutDepth))

       for downY in range(OutV):
           for acrossX in range(OutV):
               V3[downY,acrossX,0]=np.max(X1[S*downY:S*downY+F1,S*acrossX:S*acrossX+F1,:])
       return V3

def makeLayer(In:int,Out:int)->np.ndarray:
        return np.random.normal(0,1,(In,Out))

def makeHidden(signals:np.ndarray,weights:np.ndarray)->np.ndarray:
        res:npndarray=np.dot(weights,signals)
        res_activ=np.zeros((res.shape[0],res.shape[1]))

        i=0
        for row in res:
                for elem in row:
                        res_activ[i,0]=relu(elem)
                i+=1        

        return res_activ                
def calcOutGradients(last_layer_res:np.ndarray,targets:np.ndarray)->np.ndarray:
        gradients=np.zeros((last_layer_res.shape[0],1))

def feedForward(X:np.ndarray,layer1:np.ndarray,layer2:np.ndarray)->np.ndarray:
       res_conv=Output1(X,W0)
       res_output=Output2(res_conv)
       res_maxpooling=Maxpooling(res_output,W1,2)
     
       signals_conv=np.array([res_maxpooling.flatten()]).T
       print(signals_conv)

       res_layer1=makeHidden(signals_conv,layer1)
       print(res_layer1)
       res_layer2=makeHidden(res_layer1,layer2)
                             
      # res_maxpooling=Maxpooling(res_maxpooling,W1,2)
       return res_layer2

        
def train(X:np.ndarray,Y:np.ndarray):
   layer1=makeLayer(3,4)
   layer2=makeLayer(2,3)

   cnn_out_res=feedForward(X,layer1,layer2)
   
try:
      print(main())
       
except Exception as e:
       t.format_exc(e)


       


              
       
       
    
    

