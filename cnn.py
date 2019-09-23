import numpy as np
import traceback as t
import sys

from os import listdir
"""
In program we try to learn our Conv Neuro Net(CNN)
to determe images of squares and circles, that are stored
as row bytes(made it in Gimp).784 bytes per image with gray gradation.
Example of files:
circle_1_0.ari,square_10_1.ari
The last number is the class of figure
 
Denote.
a_0 - is singe matrix where we write input image to convulate it with...
w_l - matrix and to get result as...
a_l - matrix, which we activate with relu function and get
a_l_act - matrix. 
"""

class CNN:
        def __init__(self):              
                self.firstFCNLayer=self.makeLayer(3,225)
                self.secondFCNLayer=self.makeLayer(2,3)
                # e1 and e2 is non-activated neuron state on proper layer
                self.e1=None
                
                self.e2=None
                # here we write result of conv process 
                self.a_l=None
                # here we activate matrix a_l
                self.a_l_act=None

                self.grads_on_fraim_conv=None
                # hidden1 and hidden2 -  here we activate layer neurons row
                self.hidden1=None
               
                self.hidden2=None
                
                # sygnals from CNN to FCNN
                self.signals_conv=None

                # learning rate
                self.l_r=0.07
                # in general it will be 3-d array (20,1,784)
                # As mnist data.As vectors (One row in matrix)
                self.CircleAndSquaresData=[]
                # in general it will be 3-d array (20,28,28)
                # As matrix.We use it in fit function 
                self.image_storage=[]
                # in general it will be 3-d array (20,1,2)
                # We use it in fit function
                self.truth_storage=[]

                self.matr_to_backprop_conv=None
                
                self.conv_params={'stride':2,'convolution':True,'center_w_l':(0,0)}
                # this is fraim to move over input image
                self.w_l=np.array([[-1,-1,-1],
                                   [-1,8,-1],
                                   [-1,-1,-1]])
                # this are indexes for fraim to move over input image
                self.indexes_a, self.indexes_b = self.create_indexes(size_axis=self.w_l.shape, center_w_l=self.conv_params['center_w_l'])


        def makeCircleAndSquaresImgData(self,dir_:str):
                files=listdir(dir_)
                byte_list:bytes=b''
               
                truth_relat:int=0 # index for one-hot encoding
                for file_nam in files:
                        # get file name from pointed dir as file_nam
                        # read it content
                        with open(dir_+file_nam,'rb') as f:
                                byte_list=f.read()
                        im_single=[[]]
                        truth_single=[[0,0]]
                      
                        truth_relat:int=int(file_nam[-5])
                        # one-hot encoding 
                        truth_single[0][truth_relat]=1.0
                       
                        for b in byte_list:
                             im_single[0].append(float(b/255.0))

                        self.CircleAndSquaresData.append(im_single)
                        self.truth_storage.append(truth_single)
              

        def makeImageStorage(self):
            for input_image in self.CircleAndSquaresData:
                self.image_storage.append(self.vector2matrix(input_image,(28,28))) # (784,1) -> (28,28)

        def create_axis_indexes(self,size_axis:int,center_w_l:int)->list:
                coords=[]
                for i in range(-center_w_l,size_axis-center_w_l):
                        coords.append(i)
                return coords        
        def create_indexes(self,size_axis:tuple,center_w_l:tuple)->tuple:
                coords_a=self.create_axis_indexes(size_axis[0],center_w_l[0])
                coords_b=self.create_axis_indexes(size_axis[1],center_w_l[0])
                return (coords_a,coords_b)

        def make_convulat_or_corelat(self,matrix,#:matrix<R>
                                     fraim,#:matrix<R>
                                     coords_for_fraim:tuple,
                                     S:int,# Stride for fraim
                                     g_val_conv_or_corelat:int):#->matrix<R> matrix_res, we have got result after conv process
              
                matrix_height:int=matrix.shape[0]
                matrix_width:int=matrix.shape[1]
                indexes_a,indexes_b=coords_for_fraim

              
                matrix_res=np.zeros((1,1))

                matrix_res_height=matrix_res.shape[0]
                matrix_res_width=matrix_res.shape[1]
                
                for i in range(matrix_height):
                        for j in range(matrix_width):
                                result = 0
                                element_exists = False
                                for a in indexes_a:
                                        for b in indexes_b:
                                                
                                                if i*S - g_val_conv_or_corelat*a >= 0 and j*S - g_val_conv_or_corelat*b >= 0 \
                                                and i*S - g_val_conv_or_corelat*a < matrix_height and j*S - g_val_conv_or_corelat*b < matrix_width:
                                                        result += matrix[i*S - g_val_conv_or_corelat*a][j*S - g_val_conv_or_corelat*b] * fraim[indexes_a.index(a)][indexes_b.index(b)]
                                                     
                                                        element_exists = True
                               
                                if element_exists:
                                        if i >= matrix_res_height:
                                                
                                                matrix_res = np.vstack((matrix_res, np.zeros(matrix_res_width)))
                                        if j >= matrix_res_width:
                                                
                                                matrix_res = np.hstack((matrix_res, np.zeros((matrix_res_height,1))))
                                        matrix_res[i][j] = result
                                        matrix_res_height=matrix_res.shape[0]
                                        matrix_res_width=matrix_res.shape[1]
                                        
                                        
                return matrix_res              
        

        def conv_feed_get_a_l(self,a_0,#:matrix<R>
                              w_l,#:matrix<R>
                              conv_params:dict,
                              ): #->matrix<R> (a_l)
                S = conv_params['stride']
                
                if conv_params['convolution']:
                        g = 1 
                else:
                        g = -1 

                return self.make_convulat_or_corelat(a_0,w_l,(self.indexes_a,self.indexes_b),S,g)        
                
        # activate a_l matrix (we have got it after conv process).This matrix called a_l_act                
        def conv_feed_get_a_l_act(self,a_l,#:matrix<R>
                      ):#->matrix<R> a_l_act
                a_l_act=[]
               
                for row in a_l:
                        row_tmp=[]
                        for elem in row:
                            row_tmp.append(self.relu(elem))    
                        a_l_act.append(row_tmp)
                return np.array(a_l_act)        
                
        def relu(self,val:float):
                if val<0:
                    return 0
                return val

        def derivate_relu(self,val:float):
                if val<0:
                        return 0.001
                
                return 1.0


        def vector2matrix(self,vector,#:matrix<R> (1,x) np.ndarray
                          matrix_shape:tuple): #->matrix<R> np.ndarray
               
               return np.reshape(vector,matrix_shape)
        
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
                        i+=1        
               
                return cost_gradients
        def updMatrixFCN(self,layer:np.ndarray,gradients:np.ndarray,enteredVal:np.ndarray)->np.ndarray:
             layerNew=layer+self.l_r*gradients*enteredVal.T
             return layerNew

        def updMatrixCNN(self,matrix,#:matrix<R>
                              delta_matrix #:matrix<R>
                         ): #->matrix<R> new have updated

                newMatr=np.zeros((matrix.shape[0],matrix.shape[1]))

                newMatr=matrix+delta_matrix

                return newMatr

              
        def feedForward(self,a_0:np.ndarray)->np.ndarray:
               self.a_l=self.conv_feed_get_a_l(a_0,self.w_l,self.conv_params)
             
               self.a_l_act=self.conv_feed_get_a_l_act(self.a_l)

               self.signals_conv=np.array([self.a_l_act.flatten()]).T
             
               self.e1,self.hidden1=self.makeHidden(self.signals_conv,self.firstFCNLayer)
               self.e2,self.hidden2=self.makeHidden(self.hidden1,self.secondFCNLayer)
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
         
           self.matr_to_backprop_conv=\
           self.vector2matrix(grads1,(15,15))
         
           self.grads_on_fraim_conv= \
           self.make_convulat_or_corelat(self.matr_to_backprop_conv,self.w_l,(self.indexes_a,self.indexes_b),1,1)
          
           self.a_l=self.updMatrixCNN(self.a_l,self.grads_on_fraim_conv)
           return self.mse(Y.T-cnn_out_res)

        def fit(self,nEpochs:int,l_r:float)->None:
          
                self.l_r=l_r
                ep=0
                while(ep<nEpochs):
                   for i in range(20):
                        cur_img=np.array(self.image_storage[i])
                        cur_truth=np.array(self.truth_storage[i])
               
                        show_mse:float=self.train(cur_img,cur_truth) 
                        if ep%10==0:
                                print('--------------------')
                                print("Error mse:",show_mse)
                                print('--------------------')
                   ep+=1
                print("grads on fraim conv shape",np.array(self.grads_on_fraim_conv).shape)
                   

           
           
#==========================================   
try:
    cnn=CNN()
    cnn.makeCircleAndSquaresImgData('./img/')
    cnn.makeImageStorage()
    cnn.fit(30,0.07)
except Exception as e:
     
       t.print_exc(file=sys.stdout)
#=========================================       


       


              
       
       
    
    

