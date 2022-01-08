from phe import paillier
import gmpy2
import pandas as pd
import numpy as np
#from paillier import *

class CSP:

    def __init__(self):
        self.public_key, self.__private_key = paillier.generate_paillier_keypair()


class MLE:
    Atag=[]
    btag=[]
    def __init__(self,A_tag,b_tag,pk):
        self.Atag=A_tag
        self.btag=b_tag
        self.pk=pk
    def calcAandb(self,col,c_users,lamda):
        Mat_A=[]
        keyring = paillier.PaillierPrivateKeyring()
        enc_lammda=self.pk.encrypt(lamda)
        for i in range(0,col):
            vec_Ak=[]
            for j in range(0,col):
                sum_t=0
                for d in range(0,c_users):
                    sum_t=sum_t + self.Atag[d][i][j]
                if i==j:
                    vec_Ak.append(sum_t +enc_lammda)
                else:
                    vec_Ak.append(sum_t)
            Mat_A.append(vec_Ak)
                        

        return Mat_A
        
                    
                        
                    
                    
                    
                    
            
        
        



def startProtocol():
        my_csp = CSP()
        #df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
        #print(df)
        
        df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names=["sepal length","sepal width","petal length","petal width","Class"])

        event_dictionary ={'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}
  
        df['y'] = df['Class'].map(event_dictionary)

        A_i=0
        b_i=0
        col=4 # thecols of the df
        m=30 # the number of user's Doi's
        D_list = []
        for i in range(0,150,5): #compute the Doi's unputs
            for j in range(i,i+5):
                x=df.iloc[j,0:4]
                x= np.asmatrix(x)
                y_i=int(df.iloc[j,-1])
                b_i=b_i + np.dot(x,y_i)
                x_t=np.transpose(x)
                A_i=A_i+np.dot(x_t,x)
            
            Do = {i: df.iloc[i:i+5, 0:4],'y': df.iloc[i:i+5,-1] , 'Ak' : A_i , 'bi':b_i}
            D_list.append(Do)
        
        pk=my_csp.public_key
        
        A_tag_to_MLE=[]
        for d in range(0,m):
            Mat_Ak=[]
            for i in range(0,col):
                vec_Ak=[]
                for j in range(0,col):
                    if j < i:
                        temp=D_list[d]['Ak'][i,j]
                        vec_Ak.append(temp)
                    else: 
                      temp=D_list[d]['Ak'][i,j]
                      vec_Ak.append(pk.encrypt(temp))
                Mat_Ak.append(vec_Ak)
            D_list[d]['Ak_tag']=Mat_Ak
            A_tag_to_MLE.append(Mat_Ak)
            
        bi_tag_to_MLE=[]    
        
        for d in range(0,m): 
            vec_Ak=[]
            for i in range(0,col):
                temp=D_list[d]['bi'][0,i]
                vec_Ak.append(pk.encrypt(temp))
            D_list[d]['bi_tag']=vec_Ak
            bi_tag_to_MLE.append(vec_Ak)
        
        my_MLE = MLE(A_tag_to_MLE, bi_tag_to_MLE,pk)
        my_MLE.calcAandb(col,m,1)
    
        return 0
    

startProtocol()

        
        