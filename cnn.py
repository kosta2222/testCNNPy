import numpy as np
import traceback as t
import sys

import hashlib as hsh

from os import listdir
"""
In program we try to learn our Conv Neuro Net(CNN)
to determe images of squares and circles, that are stored
as row bytes(made it in Gimp).784 bytes per image with gray gradation.
Example of files:
circle_1_0.ari,square_10_1.ari
The last number is the class of figure
 
Denote.
input_img - is a matrix where we write input image to convulate it with...
patch_for_conv - matrix and to get result as...
featured_map - matrix, which we activate with relu function and get
featured_map_act - matrix. 

CNN - Convolutional neuro net
FCNN -Full connected neuro net

pach - is filter or kernel, it is the fraim moving over input image

a and b are the indeces for pach when it moves over the image when we conv
"""
def sigmoid(val:float)->float:

        return (1/(1+np.exp(-val)))
class CNN:
        def __init__(self):              
                self.firstFCNLayer=self.makeLayer(3,225)
                self.secondFCNLayer=self.makeLayer(2,3)
                # e1 and e2 is non-activated neuron state on proper layer
                self.e1=None
                
                self.e2=None
                # here we write result of conv process 
                self.featured_map=None
                # here we activate matrix featured_map
                self.featured_map_act=None

                self.grads_for_feature_map_layer=None
                # hidden1 and hidden2 -  here we activate layer neurons row
                self.hidden1=None
               
                self.hidden2=None
                

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
                
                self.conv_params={'stride':2,'convolution':True,'center_patch_for_conv':(0,0)}
                # this is patch to move over input image
                self.patch_for_conv=np.array([[-1,-1,-1],
                                   [-1,8,-1],
                                   [-1,-1,-1]])
                # this are indexes for patch to move over input image
                self.indexes_a, self.indexes_b = self.create_indeces_for_patch(size_axis=self.patch_for_conv.shape, center_patch_for_conv=self.conv_params['center_patch_for_conv'])


        def makeCircleAndSquaresImgData(self,dir_:str):
                files=listdir(dir_)                
               
                truth_relat:int=0 # index for one-hot encoding
                for file_nam in files:
                        # get file name from pointed dir as file_nam
                        # read it content
                        byte_list:bytes=b''
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
                self.image_storage.append(self.vector2matrix(input_image,(28,28))) # (1,784) -> (28,28)

        def create_axis_indexes_for_patch(self,size_axis:int,center_patch_for_conv:int)->list:
                coords=[]
                for i in range(-center_patch_for_conv,size_axis-center_patch_for_conv):
                        coords.append(i)
                return coords        
        def create_indeces_for_patch(self,size_axis:tuple,center_patch_for_conv:tuple)->tuple:
                coords_a=self.create_axis_indexes_for_patch(size_axis[0],center_patch_for_conv[0])
                coords_b=self.create_axis_indexes_for_patch(size_axis[1],center_patch_for_conv[0])
                return (coords_a,coords_b)

        def make_convulat_or_corelat(self,matrix,#:matrix<R>
                                     patch,#:matrix<R>
                                     coords_for_patch:tuple,
                                     S:int,# Stride for patch
                                     g_val_conv_or_corelat:int):#->matrix<R> matrix_res, we have got result after conv process
              
                matrix_height:int=matrix.shape[0]
                matrix_width:int=matrix.shape[1]
                indexes_a,indexes_b=coords_for_patch            
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
                                                        result += matrix[i*S - g_val_conv_or_corelat*a][j*S - g_val_conv_or_corelat*b] * patch[indexes_a.index(a)][indexes_b.index(b)]                                                     
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
        

        def conv2d(self,input_img,#:matrix<R>
                              patch_for_conv,#:matrix<R>
                              conv_params:dict,
                              ): #->matrix<R> (featured_map)
                S = conv_params['stride']
                
                if conv_params['convolution']:
                        g = 1 
                else:
                        g = -1 

                return self.make_convulat_or_corelat(input_img,patch_for_conv,(self.indexes_a,self.indexes_b),S,g)        
                
        # activate featured_map matrix (we have got it after conv process).This matrix called featured_map_act                
        def conv2d_act(self,featured_map,#:matrix<R>
                      ):#->matrix<R> featured_map_act
                """
                featured_map_act=np.zeros((featured_map.shape[0],featured_map.shape[1]))

                print('in c_a featured map',self.show_matrix_hash(featured_map))
              
                for row in featured_map:
                        i=j=0
                        for elem in row:
                            featured_map_act[i][j]=self.relu(elem)    
                            j+=1
                        i+=1
                       
                
                print('in c_a featured map act',self.show_matrix_hash(featured_map_act))
                
                return featured_map_act
                """
                return sigmoid(featured_map)
                

        def derivate_sigmoid(self,val:float)->float:
                return val*(1-val)
                
        def relu(self,val:float)->float:
                if val<0:
                    return 0
                return val

        def derivate_relu(self,val:float)->float:
                if val<0:
                        return 0.001
                
                return 1.0


        def vector2matrix(self,vector,#:matrix<R> (1,x) np.ndarray
                          matrix_shape:tuple): #->matrix<R> np.ndarray
               
               return np.reshape(vector,matrix_shape)
        
        def makeLayer(self,In:int,Out:int)->np.ndarray:
                return np.random.normal(0.0,2**-0.5,(In,Out))
        
        def makeHidden(self,signals:np.ndarray,weights:np.ndarray)->(np.ndarray,np.ndarray):
                print('in m_H signals %s weights %s'%(self.show_matrix_hash(signals,weights)[0],self.show_matrix_hash(signals,weights)[1]))
                cost=np.dot(weights,signals)
                print('in m_H cost',self.show_matrix_hash(cost))
                """
                cost_activ=np.zeros((cost.shape[0],cost.shape[1]))

                i=0
                for row in cost:
                        for elem in row:
                                cost_activ[i,0]=sigmoid(elem)
                        i+=1        

                return (cost,cost_activ)
                """
                return (cost,sigmoid(cost))
        def calcOutGradientsFCN(self,e:np.ndarray,_targets:np.ndarray)->np.ndarray:
                """
                gradients=np.zeros((e.shape[0],1))
                i=0
                targets=_targets.T
                for row in e:
                        for elem in row:
                               
                              gradients[i,0]= (targets[i,0]-elem)* self.derivate_sigmoid(elem)
                             
                                             
                        i+=1
                return gradients.T
                """
                print("in c_o_G grads",((_targets - e)*self.derivate_sigmoid(e)).T.shape)
                return ((_targets.T - e)*self.derivate_sigmoid(e)).T
        def calcHidGradientsFCN(self,layer:np.ndarray,e_:np.ndarray,gradients:np.ndarray)->np.ndarray:
                #cur_gradients=np.zeros((1,layer.shape[1]))
                
                cost_gradients=np.dot(gradients,layer)
                """
                i=0
                for row in e_:
                        for elem in row:
                                cost_gradients[0,i]=cost_gradients[0,i]*self.derivate_sigmoid(elem)
                        i+=1        
               
                return cost_gradients
                """
                print("in c_H_G cost grads",np.shape(cost_gradients))
                return cost_gradients
        def updMatrixFCN(self,layer:np.ndarray,gradients:np.ndarray,enteredVal:np.ndarray)->np.ndarray:
             layerNew=layer+self.l_r*gradients*enteredVal.T
             return layerNew

        def updMatrixCNN(self,matrix,#:matrix<R>

                         delta_matrix #:matrix<R>
                         ): #->matrix<R> new have updated

                newMatr=matrix+delta_matrix

                return newMatr

              
        def feedForward(self,input_img:np.ndarray,patch_for_conv)->np.ndarray:
               self.featured_map=self.conv2d(input_img,patch_for_conv,self.conv_params)
               print('in f_F input img %s and patch %s'%(self.show_matrix_hash(input_img),self.show_matrix_hash(self.patch_for_conv)))
             
               self.featured_map_act=self.conv2d_act(self.featured_map)
               print('in f_F feat map ',self.show_matrix_hash(self.featured_map))
               print('in f_F feat map act ',self.show_matrix_hash(self.featured_map_act))
               signals_from_CNN_to_FCNN=np.array([self.featured_map_act.flatten()]).T

               print('in f_F flat',self.show_matrix_hash(signals_from_CNN_to_FCNN))
               self.e1,hidden1=self.makeHidden(signals_from_CNN_to_FCNN,self.firstFCNLayer)
               print('in f_F hidden1',self.show_matrix_hash(self.hidden1))
               self.e2,hidden2=self.makeHidden(hidden1,self.secondFCNLayer)
               print('in f_F hidden2',self.show_matrix_hash(self.hidden2))
               return hidden2

        def show_matrix_hash(self,*matr)->str:
                hash_obj=hsh.sha256()
                matr_list=['']*len(matr)
                j=0
                #:matrix<R> as str
                for i  \
                    in matr:   
                         hash_obj.update(str(matr).encode('ascii'))
                         matr_list[j]=hash_obj.hexdigest()
                         j+=1
                return matr_list        
        def mse(self,vec:np.ndarray)->float:
                return np.square(vec).mean(axis=0)
       
        def train(self,X:np.ndarray,Y:np.ndarray)->float:

           wholeNN_out=self.feedForward(X,self.patch_for_conv)
           print('cnn out res',wholeNN_out)
        
           # Now we make backpropaganation!
           out_grads=self.calcOutGradientsFCN(wholeNN_out,Y)
          
           grads2=self.calcHidGradientsFCN(self.secondFCNLayer,self.e2,out_grads)
           
         
           #self.secondFCNLayer=self.updMatrixFCN(self.secondFCNLayer,grads2,self.hidden1)
           grads1=self.calcHidGradientsFCN(self.firstFCNLayer,self.e1,grads2)
         
           #self.firstFCNLayer=self.updMatrixFCN(self.firstFCNLayer,grads1,self.signals_from_CNN_to_FCNN)
           
           grads_from_FCNN_as_matrix= \
           self.vector2matrix(grads1,(15,15))
           # Some important steps of backpropaganation of conv are skipped
          
           grads_for_kernel= \
           self.make_convulat_or_corelat(X,grads_from_FCNN_as_matrix,self.create_indeces_for_patch(grads_from_FCNN_as_matrix.shape,(0,0)),S=10,g_val_conv_or_corelat=-1)
           
           # Update paches(kernels) matrix data!
           # self.patch_for_conv= self.updMatrixCNN(self.patch_for_conv,grads_for_kernel)
           self.patch_for_conv=self.patch_for_conv+grads_for_kernel
           print('in f_F grads for kernel',self.show_matrix_hash(grads_for_kernel))
           print('in f_F patch for conv down',self.show_matrix_hash(self.patch_for_conv))
           
           return self.mse(Y.T-wholeNN_out)
           
        def fit(self,nEpochs:int,l_r:float)->None:
          
                self.l_r=l_r
                ep=0
                while(ep<nEpochs):
                   for i in range(20):
                        cur_img=np.array(self.image_storage[i])
                                              
                        cur_truth=np.array(self.truth_storage[i])
                      
                        show_mse:float=self.train(cur_img,cur_truth) 
                        #if ep%1==0:
                        print('--------------------')
                        print("Error mse:",show_mse)
                        print('--------------------')
                   ep+=1
               
                

           
           
#==========================================   
try:
    cnn=CNN()
    cnn.makeCircleAndSquaresImgData('./img/')
    cnn.makeImageStorage()
    cnn.fit(30,0.07)
except Exception as e:
     
       t.print_exc(file=sys.stdout)
#=========================================       


       


              
       
       
    
    

