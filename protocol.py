from phe import paillier
import gmpy2
import pandas as pd
import numpy as np

from sklearn import linear_model
from numpy.linalg import inv
from simplefractions import simplest_from_float

def column(matrix, i):
    return [row[i] for row in matrix]


class CSP:

    def __init__(self):
        self.public_key, self.__private_key = paillier.generate_paillier_keypair()

    def calc_CD(self,C,d,col,c_users):
        Mat_C=[]
        for i in range(0,col): #create C tag
            vec=[]
            for j in range(0,col):
                t=self.__private_key.decrypt(C[i][j])
                vec.append(t)
            Mat_C.append(vec)
        vec_dtag=[]
        for i in range(0,col):
            t=self.__private_key.decrypt(d[i])
            vec_dtag.append(t)
        
        C_new = np.array(Mat_C)  
        C_inverse= inv(C_new)
        
        d_new= np.array(vec_dtag)
        
        w= C_inverse.dot(d_new)
        w_tag=[]
        for i in range(0,col):
            w_tag.append(w[i]% 2048)

        return w_tag
        
    
                
                
        

class MLE:
    Atag=[]
    btag=[]
    Mat_A_MLE=[]
    vec_b_MLE=[]
    R_matrix = []
    r_vec =[]
    def __init__(self,A_tag,b_tag,pk):
        self.Atag=A_tag
        self.btag=b_tag
        self.pk=pk
    
    def calcAandb(self,col,c_users,lamda):
        #Mat_A=[]
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
            self.Mat_A_MLE.append(vec_Ak)
        #vec_b=[]
        for i in range(0,col):
            sum_t=0
            for d in range(0,c_users):
                sum_t=sum_t + self.btag[d][i]
            self.vec_b_MLE.append(sum_t)
         
               
    
    def DataMasking(self, cols): #step1_prot3
        C_tag = []
        D_tag = []
        
        isinv = 0   
        R_matrix = np.random.randint(1,2048,size = (cols,cols))
        Rinv =  inv(R_matrix)
        if(np.array_equal(np.matmul(R_matrix, Rinv) , np.matmul(Rinv, R_matrix))):
              isinv = 1
        while(isinv == 1):
           isinv =0;
           R_matrix = np.random.randint(1,2048,size = (cols,cols))
           if(np.array_equal(np.matmul(R_matrix, Rinv) , np.matmul(Rinv, R_matrix))):
                 isinv = 1
        r_vec = np.random.randint(1,2048,size = (cols))
       
        self.R_matrix = R_matrix
        self.r_vec = r_vec
        
        for i in range(0, cols):
            tmp1 =0
            vect_to_append = []
            for j in range(0, cols):
                tmp = 0
                for k in range(0, cols):
                    tmp += R_matrix[k,j]* self.Mat_A_MLE[i][k]
                vect_to_append.append(tmp) 
            C_tag.append(vect_to_append)
            for k in range(0, cols):
                tmp1 += r_vec[k]*self.Mat_A_MLE[i][k]
            D_tag.append(self.vec_b_MLE[i]+tmp1)
        return C_tag, D_tag
    
    def laststep(self,w_tag,col):
        Rw_=[]
        for i in range(0,col):
            tmp2=0
            for j in range(0,col):
                tmp2 += self.R_matrix[i][j]*w_tag[j]
            Rw_.append(tmp2)
        for i in range(0,col):
            wWithoutMod = Rw_ - self.r_vec
        endW = []
        for i in range(0,col):
            endW.append(wWithoutMod[i]% 2048)
        last_=[]
        for i in range(0,col):
            last_.append(simplest_from_float(endW[i]))
        return endW
            
        
        
        
                    
                    
                    
                    
            
        
        



def startProtocol():
        my_csp = CSP()
        #df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
        #print(df)
        
        df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names=["sepal length","sepal width","petal length","petal width","Class"])

        event_dictionary ={'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}
  
        df['y'] = df['Class'].map(event_dictionary)

        reg= linear_model.LinearRegression(fit_intercept=False)
        
        reg.fit(df[['sepal length','sepal width','petal length','petal width']],df.y)

        
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
        w =startprotocol2(my_MLE,my_csp,col,m)
        
        df_t =df.pop('Class')
        df_t =df.pop('y')
        matttt=df.to_numpy()
        
        pred_y= matttt.dot(w)
        ex_df = pd.DataFrame({ 'Y without enc' :pred_y})
        ex_df.to_excel('iris.xlsx')
    
        return 0
    
def startprotocol2(my_MLE,my_CSP,col,c_users):
    
    C_tag, D_tag = my_MLE.DataMasking(col)
    w = my_CSP.calc_CD(C_tag,D_tag,col,c_users)
    w =my_MLE.laststep(w, col)
    return w
    
    
    
startProtocol()

        
        