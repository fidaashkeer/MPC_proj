from phe import paillier
import gmpy2
import pandas as pd
import numpy as np

class CSP:

    def __init__(self):
        public_key, private_key = paillier.generate_paillier_keypair()
    def ret_pk(self):
        return self.public_key()



def startProtocol():
        my_csp = CSP()
        #df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
        #print(df)
        
        df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names=["sepal length","sepal width","petal length","petal width","Class"])

        event_dictionary ={'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}
  
        df['y'] = df['Class'].map(event_dictionary)

        A_i=0
        m=30 # the number of user's Doi's
        D_list = []
        for i in range(0,150,5): # 
            
            for j in range(i,i+5):
                x=df.iloc[j,0:4]
                x= np.asmatrix(x)
                x_t=np.transpose(x)
                A_i=A_i+np.dot(x_t,x)
            
            Do = {i: df.iloc[i:i+5, 0:4],'y': df.iloc[i:i+5,-1] , 'Ak' : A_i}
            D_list.append(Do)
            
                
        
        return 0
    

startProtocol()

        
        