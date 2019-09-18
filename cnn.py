import numpy as np
import traceback as t
import sys

from os import listdir

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

                self.ko_data=[]

                self.image_starage=[]
                self.truth_storage=[]

        def makeKoData(dir_:str):
                files=listdir(_dir)
                byte_list:bytes=b''
                im=[[]]
                
                for file_nam in files:
                        with open(dir_+file_nam) as f:
                                byte_list=f.read()
         
                        for b in byte_list:
                             im[0].append(float(b/255))

                        self.ko_data.append(im)     



        def create_axis_indexes(self,size_axis:int,center_w_l:int)->list:
                coords=[]
                for i in range(-center_w_l,size_axis-center_w_l):
                        coords.append(i)
                return coords        
        def create_indexes(self,size_axis:tuple,centre_w_l:tuple)->tuple:
                coords_a=self.create_axis_indexes(size_axis[0],center_w_l[0])
                coords_b=self.create_axis_indexes(size_axis[1],center_w_l[0])
                return (coords_a,coords_b)

        def conv_feed_get_a_l(self,a_0,#:matrix<R>
                              w_l,#:matrix<R>
                              conv_params:dict,
                              ): #->matrix<R> (a_l)
                indexes_a, indexes_b = create_indexes(size_axis=w_l.shape, center_w_l=conv_params['center_w_l'])
                stride = conv_params['stride']
                # матрица выхода будет расширяться по мере добавления новых элементов
                a_l = np.zeros((1,1))
                # в зависимости от типа операции меняется основная формула функции
                if conv_params['convolution']:
                        g = 1 # операция конволюции
                else:
                        g = -1 # операция корреляции
                # итерация по i и j входной матрицы a_0 из предположения, что размерность выходной матрицы a_l будет такой же
                for i in range(a_0.shape[0]): # 
                        for j in range(a_0.shape[1]):
                                demo = np.zeros([a_0.shape[0], a_0.shape[1]]) # матрица для демонстрации конволюции
                                result = 0
                                element_exists = False
                                for a in indexes_a:
                                        for b in indexes_b:
                                                # проверка, чтобы значения индексов не выходили за границы
                                                if i*stride - g*a >= 0 and j*stride - g*b >= 0 \
                                                and i*stride - g*a < a_0.shape[0] and j*stride - g*b < a_0.shape[1]:
                                                        result += a_0[i*stride - g*a][j*stride - g*b] * w_l[indexes_a.index(a)][indexes_b.index(b)] # перевод индексов в "нормальные" для извлечения элементов из матрицы w_l
                                                        demo[i*stride - g*a][j*stride - g*b] = w_l[indexes_a.index(a)][indexes_b.index(b)]
                                                        element_exists = True
                                # запись полученных результатов только в том случае, если для данных i и j были произведены вычисления
                                if element_exists:
                                        if i >= a_l.shape[0]:
                                                # добавление строки, если не существует
                                                a_l = np.vstack((a_l, np.zeros(a_l.shape[1])))
                                        if j >= a_l.shape[1]:
                                                # добавление столбца, если не существует
                                                a_l = np.hstack((a_l, np.zeros((a_l.shape[0],1))))
                                        a_l[i][j] = result
                                        # вывод матрицы demo для отслеживания хода свертки
                                        # print('i=' + str(i) + '; j=' + str(j) + '\n', demo)
                return a_l
                
                        
                               
        def relu(self,val:float):
                if val<0:
                    return 0
                return val

        def derivate_relu(self,val:float):
                if val<0:
                        return 0.001
                
                return 1.0


        """
        Уменьшаем размер матрицы сигналов(это отдаем),слоя к которому мы применяем
        окошко весов-так мы выделяем признаки
        """
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
        """
        Выясняем активированное состояние нейронов сверточного слоя(отдаем)
        """      
        def Conv_act(self,V1:np.ndarray)->np.ndarray:
               V2=np.zeros((V1.shape[0],V1.shape[0],V1.shape[2]))

               for row in range(V1.shape[0]):
                   for elem in range(V1.shape[1]):
                       V2[row][elem][0]=self.relu(V1[row][elem][0])
               return V2
        """
        Применяем другое окошко для макс-пулинга чтобы пройтись по
        матрице после свертки/активации(отдаем)
        """  
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
        """
        Заинициализировали 1 слой FCNN сети
        """  
        def makeLayer(self,In:int,Out:int)->np.ndarray:
                return np.random.normal(0,1,(In,Out))
        """
        Нужно для прямого распространения,взвешиваем сигналы на слое,
        отдаем взвешинные сигналы и их же пропущенных через активационную функцию
        """ 
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
             
               self.e1,self.hidden1=self.makeHidden(self.signals_conv,self.firstFCNLayer)
               self.e2,self.hidden2=self.makeHidden(self.hidden1,self.secondFCNLayer)
               return self.hidden2
        """
        Средне-квадратичная ошибка
        """
        def mse(self,vec:np.ndarray)->float:
                return np.square(vec).mean(axis=0)
        # Один раз проганяем сигнал, один раз обновляем,показываем при этом средне-квадратичную ошибку 
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

           self.res_maxpooling:np.ndarray=self.updMatrixCNNMaxpooling(self.res_maxpooling,grads1)

           return self.mse(Y-cnn_out_res)

        def fit(self,WholeMatrix:np.ndarray,nEpochs:int,l_r:float)->None:
           self.l_r=l_r
           ep=0
           while(ep<nEpochs):
                for X,Y in WholeMatrix:
                        show_mse:float=self.train(X,Y)
                        if ep%1000==0:
                                print("Error mse:",show_mse)
                   

           
           
#==========================================   
try:
    cnn=CNN()
    cnn.train(cnn.X,np.array([[0,1]]))
except Exception as e:
      # with open('log','w') as f: 
       #  t.print_exc(file=f)
       t.print_exc(file=sys.stdout)
#=========================================       


       


              
       
       
    
    

