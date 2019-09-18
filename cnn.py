import numpy as np
import traceback as t
import sys

from os import listdir

class CNN:
        def __init__(self):              
                self.firstFCNLayer=self.makeLayer(3,225)
                self.secondFCNLayer=self.makeLayer(2,3)
                # e неактивированное состояние нейронов на слоях FCNN
                self.e1=None
                # 
                self.e2=None
                # 
                self.a_l=None
                # 
                self.a_l_act=None
                
                # 
                self.hidden1=None
                # 
                self.hidden2=None
                
                # сигналы с CNN на FCNN
                self.signals_conv=None

                
                
                self.l_r=0.07

                self.ko_data=[]

                self.image_storage=[]
                self.truth_storage=[]
                
                self.conv_params={'stride':2,'convolution':True,'center_w_l':(0,0)}
                self.w_l=np.array([[-1,-1,-1],
                                   [-1,8,-1],
                                   [-1,-1,-1]]) 

        def makeKoData(self,dir_:str):
                files=listdir(dir_)
                byte_list:bytes=b''
                im=[[]]
                
                for file_nam in files:
                        with open(dir_+file_nam,'rb') as f:
                                byte_list=f.read()
                        im[0].clear()      
                        for b in byte_list:
                             im[0].append(float(b/255.0))

                        self.ko_data.append(im)

        def makeImageStorage(self):
            for input_image in self.ko_data:
            	self.image_storage.append(np.reshape(input_image, (28, 28))) # (784,1) -> (28,28)



        def create_axis_indexes(self,size_axis:int,center_w_l:int)->list:
                coords=[]
                for i in range(-center_w_l,size_axis-center_w_l):
                        coords.append(i)
                return coords        
        def create_indexes(self,size_axis:tuple,center_w_l:tuple)->tuple:
                coords_a=self.create_axis_indexes(size_axis[0],center_w_l[0])
                coords_b=self.create_axis_indexes(size_axis[1],center_w_l[0])
                return (coords_a,coords_b)

        def conv_feed_get_a_l(self,a_0,#:matrix<R>
                              w_l,#:matrix<R>
                              conv_params:dict,
                              ): #->matrix<R> (a_l)
                indexes_a, indexes_b = self.create_indexes(size_axis=w_l.shape, center_w_l=self.conv_params['center_w_l'])
                stride = self.conv_params['stride']
                
                a_l = np.zeros((1,1))
               
                if self.conv_params['convolution']:
                        g = 1 
                else:
                        g = -1 
                
                for i in range(a_0.shape[0]): # 
                        for j in range(a_0.shape[1]):
                                demo = np.zeros([a_0.shape[0], a_0.shape[1]]) 
                                result = 0
                                element_exists = False
                                for a in indexes_a:
                                        for b in indexes_b:
                                                
                                                if i*stride - g*a >= 0 and j*stride - g*b >= 0 \
                                                and i*stride - g*a < a_0.shape[0] and j*stride - g*b < a_0.shape[1]:
                                                        result += a_0[i*stride - g*a][j*stride - g*b] * w_l[indexes_a.index(a)][indexes_b.index(b)]
                                                        demo[i*stride - g*a][j*stride - g*b] = w_l[indexes_a.index(a)][indexes_b.index(b)]
                                                        element_exists = True
                               
                                if element_exists:
                                        if i >= a_l.shape[0]:
                                                
                                                a_l = np.vstack((a_l, np.zeros(a_l.shape[1])))
                                        if j >= a_l.shape[1]:
                                                
                                                a_l = np.hstack((a_l, np.zeros((a_l.shape[0],1))))
                                        a_l[i][j] = result
                                        
                                        #print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
                return a_l
                
                        
        def conv_feed_get_a_l_act(self,a_l,#:matrix<R>
                      ):#->matrix<R> a_l_act
                a_l_act_py=[]
                row_tmp=[]
                for row in a_l:
                        row_tmp.clear()
                        for elem in row:
                            row_tmp.append(self.relu(elem))    
                        a_l_act_py.append(row_tmp)
                return np.array(a_l_act_py)        
                
        def relu(self,val:float):
                if val<0:
                    return 0
                return val

        def derivate_relu(self,val:float):
                if val<0:
                        return 0.001
                
                return 1.0


        
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
                               
                              gradients[i,0]= (targets[i,0]-elem)* self.derivate_relu(elem)
                             
                                             
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
                

        def feedForward(self,a_0:np.ndarray)->np.ndarray:
               self.a_l=self.conv_feed_get_a_l(a_0,self.w_l,self.conv_params)
               print('res conv shape',self.a_l.shape)
               self.a_l_act=self.conv_feed_get_a_l_act(self.a_l)
               print('res conv act shape',self.a_l_act.shape)
	

               self.signals_conv=np.array([self.a_l_act.flatten()]).T
             
               self.e1,self.hidden1=self.makeHidden(self.signals_conv,self.firstFCNLayer)
               self.e2,self.hidden2=self.makeHidden(self.hidden1,self.secondFCNLayer)
               return self.hidden2
                  
       
        def mse(self,vec:np.ndarray)->float:
                return np.square(vec).mean(axis=0)
       
        def train(self,X:np.ndarray,Y:np.ndarray)->float:

           cnn_out_res=self.feedForward(X)
           out_grads=self.calcOutGradientsFCN(cnn_out_res,Y)
           print("out grads:",out_grads.shape)
           grads2=self.calcHidGradientsFCN(self.secondFCNLayer,self.e2,out_grads)
           print("grads on layer 2:",grads2.shape)
           self.secondFCNLayer=self.updMatrixFCN(self.secondFCNLayer,grads2,self.hidden1)
           grads1=self.calcHidGradientsFCN(self.firstFCNLayer,self.e1,grads2)
           print("grads on layer 1:",grads1.shape)
           self.firstFCNLayer=self.updMatrixFCN(self.firstFCNLayer,grads1,self.signals_conv)
           return self.mse(Y-cnn_out_res)

        def fit(self,nEpochs:int,l_r:float)->None:
          
                self.l_r=l_r
                #ep=0
                #while(ep<nEpochs):
                for X in np.array(self.image_storage):
                        show_mse:float=self.train(X,np.array([[1.0,0.0]])) # Y для теста
                        #if ep%1000==0:
                                #print("Error mse:",show_mse)
                   

           
           
#==========================================   
try:
    cnn=CNN()
    cnn.makeKoData('./img/')
    cnn.makeImageStorage()
    cnn.fit(12,0.07)
except Exception as e:
     
       t.print_exc(file=sys.stdout)
#=========================================       


       


              
       
       
    
    

