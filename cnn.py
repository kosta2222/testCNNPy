import numpy as np
import traceback as t
import sys

class CNN:
        def __init__(self):
                self.F=3
                self.F1=2
                self.W=7

                self.d=3
                self.S=1
                self.P=0
                # биас
                self.b0=+1

                np.random.seed(42)
                # тестировочный подвыборочны слой
                self.X=np.random.normal(0,1,(self.W,self.W,self.d))
                # окно для него
                self.W0=np.random.normal(0,1,(self.F,self.F,self.d))
                # окно для макс-пулинг
                self.W1=np.random.normal(0,1,(self.F1,self.F1,self.d))

                self.firstFCNLayer=self.makeLayer(3,4)
                self.secondFCNLayer=self.makeLayer(2,3)
                # неакивированное состояние первого слоя
                self.e1=None
                # неактивированное состояние второго слоя
                self.e2=None
                # тензор после свертки
                self.res_conv=None
                # его пропустили через активацию
                self.res_conv_act=None
                
                # активировали первый слой
                self.hidden1=None
                # активировали второй слой
                self.hidden2=None
                
                # тензор после макс-пулинга
                self.res_maxpooling=None
                # векторизировали тензор макс-пулинга
                self.signals_conv=None

                
                # коэффициент обучения
                self.l_r=0.07

        def relu(self,val:float):
                if val<0:
                    return 0
                else:
                    return val

        def derivate_relu(self,val:float):
                if val<0:
                        return 0.001
                else:
                        return 1.0



        def Conv(self,X:np.ndarray,W0:np.ndarray)->np.ndarray:
               W=X.shape[0]
               F=W0.shape[0]
               S=1
               P=0

               OutV=int((W-F+2*P)/S+1)
               OutDepth=1

               V1=np.zeros((OutV,OutV,OutDepth))

               for downY in range(OutV):
                   for acrossX in range(OutV):
                       V1[downY,acrossX,0]=np.sum(X[S*downY:S*downY+F,S*acrossX:S*acrossX+F,:]*W0)+self.b0

               return V1

        def Conv_act(self,V1:np.ndarray)->np.ndarray:
               V2=np.zeros((V1.shape[0],V1.shape[0],V1.shape[2]))

               for row in range(V1.shape[0]):
                   for elem in range(V1.shape[1]):
                       V2[row][elem][0]=self.relu(V1[row][elem][0])
               return V2

        def Maxpooling(self,X1:np.ndarray,W1:np.ndarray,S)->np.ndarray:
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

        def makeLayer(self,In:int,Out:int)->np.ndarray:
                return np.random.normal(0,1,(In,Out))

        def makeHidden(self,signals:np.ndarray,weights:np.ndarray)->(np.ndarray,np.ndarray):
                cost:np.ndarray=np.dot(weights,signals)
                cost_activ=np.zeros((cost.shape[0],cost.shape[1]))

                i=0
                for row in cost_activ:
                        for elem in row:
                                cost_activ[i,0]=self.relu(elem)
                        i+=1        

                return (cost,cost_activ)                
        def calcOutGradientsFCN(self,e:np.ndarray,_targets:np.ndarray)->np.ndarray:
                gradients=np.zeros((e.shape[0],1))
                i=0
                targets=_targets.T
                for row in e:
                        for elem in row:
                               
                              gradients[i,0]=(targets[i,0]-elem)* self.derivate_relu(elem)
                        i+=1
                return gradients.T
        def calcHidGradientsFCN(self,layer:np.ndarray,e_:np.ndarray,gradients:np.ndarray)->np.ndarray:
                cur_gradients=np.zeros((1,layer.shape[1]))
                cost_gradients=np.dot(gradients,layer)
                i=0
                for row in e_:
                        for elem in row:
                                cost_gradients[0,i]=cost_gradients[0,i]*self.derivate_relu(elem)
               
                return cur_gradients
        def updMatrixFCN(self,layer:np.ndarray,gradients:np.ndarray,enteredVal:np.ndarray)->np.ndarray:
             print("layer shape",layer.shape,"gradients",gradients.shape,"enteredval",enteredVal.shape)
             layerNew=layer+self.l_r*gradients*enteredVal.T
             return layerNew

        def updMatrixCNNMaxpooling(self,layer:np.ndarray,gradients:np.ndarray)->np.ndarray:
                layerNew=layer+self.l_r*gradients
                return layerNew

        def calcHidGradientsCNNMaxpooling(self,layer:np.ndarray,gradients:np.ndarray)->np.ndarray:
                cost_gradients=np.dot(gradients,layer)
                return cost_gradients
                

        def feedForward(self,X:np.ndarray)->np.ndarray:
               self.res_conv=self.Conv(self.X,self.W0)
               self.res_conv_act=self.Conv_act(self.res_conv)
               self.res_maxpooling=self.Maxpooling(self.res_conv_act,self.W1,2)

               self.signals_conv=np.array([self.res_maxpooling.flatten()]).T
              # print(signals_conv)

               self.e1,self.hidden1=self.makeHidden(self.signals_conv,self.firstFCNLayer)
               #print(res_layer1)
               self.e2,self.hidden2=self.makeHidden(self.hidden1,self.secondFCNLayer)

              # res_maxpooling=Maxpooling(res_maxpooling,W1,2)
               return self.hidden2
        def mse(self,vec:np.ndarray)->float:
                return np.square(vec).mean(axis=0)

        def train(self,X:np.ndarray,Y:np.ndarray)->float:

           cnn_out_res=self.feedForward(X)
           out_grads=self.calcOutGradientsFCN(cnn_out_res,Y)
           grads2=self.calcHidGradientsFCN(self.secondFCNLayer,self.e2,out_grads)
           self.secondFCNLayer=self.updMatrixFCN(self.secondFCNLayer,grads2,self.hidden1)
           grads1=self.calcHidGradientsFCN(self.firstFCNLayer,self.e1,grads2)
           self.firstFCNLayer=self.updMatrixFCN(self.firstFCNLayer,grads1,self.signals_conv)

           self.res_maxpooling:np.ndarray=self.updMatrixCNNMaxpooling(self.res_maxpooling,grads1)

           return self.mse(Y-cnn_out_res)

        def fit(self,WholeMatrix:np.ndarray,nEpochs:int,l_r:float)->None:
           self.l_r=l_r
           ep=0
           while(ep<nEpochs):
                   pass 
                   

           
           
#==========================================   
try:
    cnn=CNN()
    cnn.train(cnn.X,np.array([[0,1]]))
except Exception as e:
      # with open('log','w') as f: 
       #  t.print_exc(file=f)
       t.print_exc(file=sys.stdout)
#=========================================       


       


              
       
       
    
    

