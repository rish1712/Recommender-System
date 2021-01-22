""" Recommender System"""
from dataset import *
from math import sqrt
import random
class Colaborative_Filtering:
    """
    Class Collaborative_Filtering
    normal: numpy array which is a copy of the train dataset array
    cosine_similarity_maatrix: numpy array which contains the cosine similarity between the two vectors
    """
    def __init__(self):
        """
        Constructor for Collaborative Filtering class
        """
        self.normal=train_mat.copy()
        self.cosine_similarity_maatrix=np.zeros((len(train_mat),len(train_mat)))


    def normalizeMatrix(self):
        """
            Takes the normal vector and normalizes the entries present in it by using the mean of the array
        """
        length=len(self.normal)
        for i in range(length):
            mean=self.calculate_mean(self.normal[i])
            self.normal[i][self.normal[i]>0]-=mean

    def calculate_mean(self,vector):
        """
            Finds the mean of the given vector and returns the mean of the given vector
        :param vector: A numpy array whose mean has to be found
        :return: An integer which is mean of the given array
        """
        sum=np.sum(vector)
        a=vector>0
        n=np.sum(a)
        if(n==0):
            return 0
        return (sum/n)

    def cosine_similarity(self,matrix,vector,index):

        """
        This method calculates the cosine similarity between the given vector and each rows of the given matrix
        :param matrix: A 2-D numpy array which is the test dataset
        :param vector: A 1-D numpy array
        :return: 1-D numpy array containing the similarity between two vectors
        """
        temp=np.dot(matrix,vector)
        z=np.linalg.norm(vector)
        y=np.linalg.norm(matrix,axis=1)
        temp=temp.T
        return (temp/(z*y+0.000000000000000000000000000001))

    def pairwisesim(self):
        """
        This method calculates the pairwise similarity between 2 users
        :return: void
        """
        length=len(self.cosine_similarity_maatrix)
        for i in range(length):
            vector=np.array([self.normal[i]])
            self.cosine_similarity_maatrix[i]=self.cosine_similarity(self.normal,vector.T,i)



    def find_expected_ratings(self,bool_check):
        """
        This method calculates the rating of any user by the by using the test_data_list. The bool_check value tells us whether baseline approach has to be considered in collaborative
        filtering approach. If bool_check is true then baseline method is used. This method returns the predicted rating by our model
        :param bool_check: boolean
        :return: A 2-d numpy array containg the actual and predicted rating
        """
        length=len(test_data_list)
        global_avg=self.calculate_mean(train_mat)
        predicted_rating=np.zeros((2,length))
        for i in range(length):
            rating=self.find_similar_score(test_data_list[i][0],test_data_list[i][1],test_data_list[i][2],bool_check,global_avg)
            if bool_check is False:
                predicted_rating[0][i]=rating
                predicted_rating[1][i]=test_data_list[i][2]
            elif bool_check is True:
                user_avg=self.calculate_mean(train_mat[test_data_list[i][0]])
                x=train_mat.T
                movie_avg=self.calculate_mean(x[test_data_list[i][1]])
                predicted_rating[0][i]=rating+user_avg+movie_avg-global_avg
                predicted_rating[1][i] = test_data_list[i][2]

        return predicted_rating


    def find_similar_score(self,user,movie,actual_rating,bool_check,global_avg):
        """
        This method calls the find_top_k method
        :param user: user-id of an user
        :param movie: movie-id of a movie
        :param actual_rating: Actual rating according to test data
        :param bool_check: Boolean value for baseline approach
        :param global_avg: The global mean of the test data
        :return: void
        """
        return self.find_top_k(self.cosine_similarity_maatrix[user],train_mat.T[movie],actual_rating,bool_check,global_avg,movie)

    def find_top_k(self,cosine_vector,movie_vector,actual_rating,bool_check,global_avg,movie):
        """
        This method finds the top k similar users with respect to a user for a particular user
        :param cosine_vector: The cosine similarity numpy array w.r.t given user
        :param movie_vector: The ratings for a particular movie
        :param actual_rating: Actual rating according to test data
        :param bool_check: Boolean value for baseline approach
        :param global_avg: The global mean of the test data
        :param movie: movie-id of a movie
        :return:
        """
        temp=np.zeros((3,len(cosine_vector)))
        temp[0]=cosine_vector
        temp[1]=movie_vector
        temp[2]=np.arange(0,len(cosine_vector),1)
        temp=temp.T
        list1=temp.tolist()
        list1.sort(reverse=True)
        temp=np.array(list1)
        temp=temp.T
        return self.calculate_weighted_avg(temp,actual_rating,bool_check,global_avg,movie)


    def calculate_weighted_avg(self,temp,actual_rating,bool_check,global_avg,movie):
        """
        Calculates the weighted score of given matrix w.r.t to the similarity matrix
        :param temp: Temporoary numpy array storing the movie id values
        :param actual_rating: actual rating of a given user for a particular movie
        :param bool_check: boolean value for deciding whether to take baseline approach or not
        :param global_avg: The global average of the train dataset
        :param movie: movie id of a particular user
        :return:
        """
        product=0
        sum=0
        count=0
        l=temp.shape
        length=l[1]
        for i in range(1,length):
            if temp[1][i]!=0:
                sum+=temp[0][i]
                count+=1
                if bool_check is False:
                    product += temp[1][i] * temp[0][i]
                if bool_check is True:
                    user_avg = self.calculate_mean(train_mat[int(temp[2][i])])
                    x = train_mat.T
                    movie_avg = self.calculate_mean(x[movie])
                    baseline=user_avg+movie_avg-global_avg
                    product+=temp[0][i]*(temp[1][i]-baseline)

            if count>35:
                break
        if sum==0:
            return actual_rating
        return (product/sum)

class Error:
    """
    Class Error
    """
    def __init__(self):
        self.top_k=500
    def rmse(self,rating,xyz):
        """
        This method calculates the Root Mean Square Error from predicted and actual rating of the user
        :param rating: A 2-d numpy array containing the actual and predicted rating
        :param xyz: integer for deciding which model is used for calculations
        :return: void
        """
        b=rating[0]-rating[1]
        c=np.square(b)
        sum=np.sum(c)
        x=rating.shape
        rmse=(sum/x[1])
        #print(rmse)
        rmse=sqrt(rmse)
        #print(rmse)
        #print(x)
        if xyz == 4:
            print("The Root mean Square error in SVD model with 90* retained Energy is " + str(rmse))
        if xyz==1:
            print("The Root mean Square error in collaborative filtering model without baseline approach is "+str(rmse))
        if xyz==2:
            print("The Root mean Square error in collaborative filtering model with baseline approach is "+str(rmse))
        if xyz == 6:
            print("The Root mean Square error in CUR model with 90% retained Energy is " + str(rms(rmse)))
        if xyz == 5:
            print("The Root mean Square error in CUR model with 100% retained Energy is " + str(rmse))
        if xyz == 3:
            print("The Root mean Square error in SVD model with 100% retained Energy is " + str(rmse))

    def precison_at_top_k(self,rating,x):
        """
        This method calculates the Prescison at the top k from predicted and actual rating of the user
        :param rating: A 2-d numpy array containing the actual and predicted rating
        :param xyz: integer for deciding which model is used for calculations
        :return: void
        """

        list1=rating.T.tolist()
        list1.sort(reverse=True)
        rating=np.array(list1)
        rating=rating.T
        occurence_more_than_3= rating[1]>3
        no_of_relevant_items=occurence_more_than_3.sum()
        recommend_array=np.array(list1[:self.top_k])
        recommend_array=recommend_array.T
        recommend_greater_than_3=recommend_array[0]>3
        relevant_greater_than_3=recommend_array[1]>3
        no_of_recomend_items=0
        for i in range(self.top_k):
            if recommend_greater_than_3[i]==True and relevant_greater_than_3[i]==True:
                no_of_recomend_items+=1

        prescison=(no_of_recomend_items/no_of_relevant_items)
        #print("the prescison is "+str(prescison))
        if x==1:
            print("The Prescison at top K in collaborative filtering model without baseline approach is "+str(prescison))
        if x==2:
            print("The Prescison at top K in collaborative filtering model with baseline approach is "+str(prescison))
        if x == 5:
            print("The Prescison at top K in CUR model with 100% retained Energy is " + str(prescison))
        if x == 3:
            print("The Prescison at top K in SVD model with 100% retained Energy is " + str(prescison))
        if x == 6:
            print("The Prescison at top K in CUR model with 90% retained Energy is " + str(prescison))
        if x == 4:
            print("The Prescison at top K in SVD model with 90* retained Energy is " + str(prescison))


    def spearman_rank_corelation(self,rating,x):
        """
        This method calculates the Spearman Rank Correlation from predicted and actual rating of the user
        :param rating: A 2-d numpy array containing the actual and predicted rating
        :param xyz: integer for deciding which model is used for calculations
        :return: void
        """
        xyz = rating.shape
        length = xyz[1]
        ranks = np.zeros((2, length))
        ranks[0] = pd.Series(rating[0]).rank()
        ranks[1] = pd.Series(rating[1]).rank()
        mean_predicted = (np.sum(ranks[0]) / length)
        mean_actual = (np.sum(ranks[1]) / length)
        ranks[0] = ranks[0] - mean_predicted
        ranks[1] = ranks[1] - mean_actual
        temp = ranks.copy()
        temp = np.square(temp)
        sum = np.sum(temp, axis=1)
        numerator = np.dot(ranks[0], ranks[1].T)
        coff = numerator / sqrt((sum[0] * sum[1]))
        # print("the coff is "+str(coff))
        if x == 1:
            print("The Spearman Rank Correlation in collaborative filtering model without baseline approach is " + str(
                coff))
        if x == 2:
            print(
                "The Spearman Rank Correlation at top K in collaborative filtering model with baseline approach is " + str(
                    coff))
        if x == 5:
            print("The Spearman Rank Correlation at top K in CUR model with 100% retained Energy is " + str(coff))
        if x == 3:
            print("The Spearman Rank Correlation at top K in SVD model with 100% retained Energy is " + str(coff))
        if x == 6:
            print("The Spearman Rank Correlation at top K in CUR model with 90% retained Energy is " + str(coff))
        if x == 4:
            print("The Spearman Rank Correlation at top K in SVD model with 90* retained Energy is " + str(coff))

class SVD:
    """
    Class SVD
    """
    def __init__(self):
        pass
    def dimensionality_reduction(self):
        """
        This method reduces the dimensions of the SVD matrix by retaining 90% of the energy

        :return void
        """
        self.sigma=np.square(self.sigma)
        full_sum=np.sum(self.sigma)
        cnt=0
        countinous_sum=0
        length=len(self.sigma)
        for i in range(length):
            countinous_sum+=self.sigma[i][i]
            retained_energy=(countinous_sum/full_sum)
            if retained_energy >=0.9:
                cnt=i
                break
        for i in range(cnt,length):
            self.sigma[i][i]=0
        self.sigma=np.sqrt(self.sigma)


    def calculate_SVD_matrices(self,bool_val):
        """
        This method creates the U, Sigma, VT matrices of the SVD decomposition using the Eigen values of the train matrices. The bool_val tells us about whether dimensionality reduction
        has to be done or not.
        :param bool_val: boolean value
        :return: void
        """

        self.u,s,self.vt=np.linalg.svd(self.train_mat_copy,full_matrices=False)
        self.sigma=np.zeros((len(self.u),len(self.vt.T)))
        length=len(self.u)
        for i in range(length):
            self.sigma[i][i]=(s[i])
        self.vt=self.vt.T
        #print(self.sigma)
        M=np.dot(self.train_mat_copy,self.train_mat_copy.T)
        eigval,eigvec=np.linalg.eig(M)
        indexes=np.argsort(-eigval)
        u=eigvec[:,indexes]
        sigma_sq=eigval[indexes]
        sigma_sq[sigma_sq<0]=0
        sigma_sq=np.sqrt(sigma_sq)

        M = np.dot(self.train_mat_copy.T, self.train_mat_copy)
        eigval, eigvec = np.linalg.eig(M)
        indexes = np.argsort(-eigval)
        vt = eigvec[:, indexes]

        a=(u.shape)
        b=(vt.shape)

        sigma=np.zeros((a[0],b[0]))
        for i in range(a[0]):
            sigma[i][i]=sigma_sq[i]

        if bool_val is True:
            self.dimensionality_reduction()


    def normalize_data(self):
        """
        This method normalizes the given dataset bu subtracting rows from their row mean
        :return: void
        """

        self.user_offset=np.zeros(len(train_mat))
        self.train_mat_copy=full_data_set.copy()
        length=len(train_mat)
        for i in range (length):
            self.user_offset[i]=((np.sum(self.train_mat_copy[i]))/(np.sum(self.train_mat_copy[i]>0)))
            self.train_mat_copy[i][self.train_mat_copy[i]>0]-=self.user_offset[i]

    def predict_rating(self):
        """
        This method predicts the rating by using the U, Sigma, VT matrices.
        :return: A 2-d Numpy array which contains the predicted ratings
        """
        usk=np.dot(self.u,self.sigma)
        u_sigma_v=np.dot(usk,self.vt)
        #u_sigma_v=np.dot(usk,skv)
        for i in range(len(u_sigma_v)):
            u_sigma_v[i][u_sigma_v[i]!=0]+=self.user_offset[i]
        #print(u_sigma_v)
        return u_sigma_v

    def create_2d_rating(self,rating):
        """
        Creates a 2-d numpy array which contains the predicted and actual ratings
        :param rating: A 2-d numpy array which contains the predicted ratings
        :return: A 2-d numpy array which contains the actual and predicted ratings
        """
        length=np.sum(full_data_set>0)
        rating_2d=np.zeros((2,length))
        a=rating.shape
        k=0
        for i in range(a[0]):
            for j in range(a[1]):
                if full_data_set[i][j]!=0:
                    rating_2d[0][k]=rating[i][j]
                    rating_2d[1][k]=full_data_set[i][j]
                    k+=1
        return full_data_set

class CUR:
    """
    Class CUR
    """
    def __init__(self):
        pass
    def create_probabilty(self,train_mat):
        """
        This method creates the probability of all the rows of a given matrix
        :param train_mat: A 2-d numpy array
        :return: a 1-d numpy array which contains the row probabilties
        """
        train_mat=np.square(train_mat)
        sum_of_all_matrix=np.sum(train_mat)
        length=len(train_mat)
        probabilty=np.zeros(length)
        for i in range(length):
            probabilty[i]=(np.sum(train_mat[i])/sum_of_all_matrix)

        return probabilty

    def select_n_random_vectors(self,train_mat,row_probabilty,k):
        """
        This method choes k random vector based on their probabilities of occurrence
        :param train_mat: A 2-d numpy array from which rows are to be chosen randomly
        :param row_probabilty: A 1-d numpy array which stores the  probabilities of each row
        :param k: An integer which tells us how many rows has to be picked
        :return: returns 2-d numpy matrix which contains the selected rows, a 1-d numpy array which contains the indexes of the roes selected
        """
        a=np.arange(len(train_mat))
        row_index=np.random.choice(a,k,True,row_probabilty)
        #print(row_index)
        R=np.zeros((k,len(train_mat[0])))
        for i in range(k):
            R[i]=train_mat[i]
            R[i]=R[i]/float(sqrt(k*row_probabilty[i]))
        return R,row_index

    def create_intersection(self,row_indices,column_indices):
        """
        This method stores the intersection points of given array from their row and column indexes
        :param row_indices: The row indexes
        :param column_indices: The column indexes
        :return: void
        """
        self.x=np.zeros((self.k,self.k))
        for i, row in zip(range(len(row_indices)), row_indices):
            for j, column in zip(range(len(column_indices)), column_indices):
                self.x[i][j] = self.train_mat_copy[row][column]

    def dimensionality_reduction(self,eigen_values,k):
        """
        This method reduces the dimensions of given matrix according to the given 'k'
        :param eigen_values: A numpy 2-d matrix which contains whose reduction has to be done
        :param k: An integer which contains the retained energy values
        :return: A 2-d numpy array whose dimensions has been removed
        """
        eigen_values=np.square(eigen_values)
        full_sum=np.sum(eigen_values)
        cnt=0
        continous_sum=0
        for i in range(len(eigen_values)):
            continous_sum+=eigen_values[i]
            retained_energy=(continous_sum/full_sum)
            if retained_energy>=k:
                cnt=i
                break
        for i in range(cnt,len(eigen_values)):
            eigen_values[i]=0
        return eigen_values

    def create_CUR(self,bool_val):
        """
        Thise methods calculates the C, U, R matrices of a given matrix and predicts the given rating of an user
        :param bool_val: Boolean value which tells about the whether dimensions has to be reduced or not
        :return: predicted ratings of the user by CUR decomposition
        """
        self.k=300
        svd=SVD()
        svd.normalize_data()
        self.train_mat_copy=svd.train_mat_copy
        row_probabilty=self.create_probabilty(self.train_mat_copy)
        column_probabilty=self.create_probabilty(self.train_mat_copy.T)
        self.R,row_index=self.select_n_random_vectors(self.train_mat_copy,row_probabilty,self.k)
        self.C,column_index=self.select_n_random_vectors(self.train_mat_copy.T,column_probabilty,self.k)
        self.C=self.C.T
        self.create_intersection(row_index,column_index)

        X, eigen_values, YT = np.linalg.svd(self.x, full_matrices=False)
        #print(eigen_values)
        sigma = np.zeros((self.k, self.k))
        sigma_plus = np.zeros((self.k, self.k))
        if bool_val is True:

            #print(eigen_values)
            eigen_values=self.dimensionality_reduction(eigen_values,0.9)
        else:
            #print(eigen_values)
            eigen_values = self.dimensionality_reduction(eigen_values, 0.997)
        for i in range(len(eigen_values)):
            sigma[i][i] = sqrt(eigen_values[i])
            if (sigma[i][i] != 0):
                sigma_plus[i][i] = 1 / float(sigma[i][i])

        U = np.dot(np.dot(YT.T, np.dot(sigma_plus, sigma_plus)), X.T)
        # print(U)
        # CUR matrix
        cur_matrix = np.dot(np.dot(self.C, U), self.R)
        for i in range(len(cur_matrix)):
            cur_matrix[i][cur_matrix[i]!=0]+=svd.user_offset[i]
        #print(cur_matrix)
        squared_error_sum = 0
        number_of_predictions = 0

        return cur_matrix

'''

start=time.time()
print("*******************************************Collabarative Filtering without baseline approach**************************************")
c=Colaborative_Filtering()
c.normalizeMatrix()
c.pairwisesim()
rating=c.find_expected_ratings(False)
error=Error()
error.rmse(rating,1)
error.precison_at_top_k(rating,1)
error.spearman_rank_corelation(rating,1)
end=time.time()
#print(train_mat)
print("Time taken by the above model is  "+str(end-start))

start=time.time()
print("*******************************************Collabarative Filtering with baseline approach**************************************")
c=Colaborative_Filtering()
c.normalizeMatrix()
c.pairwisesim()
rating=c.find_expected_ratings(True)
error=Error()
error.rmse(rating,2)
error.precison_at_top_k(rating,2)
error.spearman_rank_corelation(rating,2)
end=time.time()
print("Time taken by the above model is  "+str(end-start))


print("********************* SVD with 100% retained energy ************************* ")
start=time.time()
svd=SVD()
svd.normalize_data()
svd.calculate_SVD_matrices(False)
#print("hi there worls")
#print(svd.u)
rating=svd.predict_rating()

reating=svd.create_2d_rating(rating)
error=Error()
error.rmse(rating,3)
error.precison_at_top_k(rating,3)
error.spearman_rank_corelation(rating,3)
end=time.time()
print("Time taken by the above model is "+str(end-start))

print("********************** SVD with 90% retained energy***************************")
start=time.time()
svd=SVD()
svd.normalize_data()
svd.calculate_SVD_matrices(True)
#print(svd.u)
rating=svd.predict_rating()
reating=svd.create_2d_rating(rating)
error=Error()
error.rmse(rating,4)
error.precison_at_top_k(rating,4)
error.spearman_rank_corelation(rating,4)
end=time.time()
print("Time taken by the above model is "+str(end-start))

print("*****************************CUR WITH 100% retained Energy***************")
start=time.time()
cur=CUR()
#cur.create_matrices()
rating=cur.create_CUR(False)
#print(rating)
svd=SVD()
reating=svd.create_2d_rating(rating)
error=Error()
error.rmse(rating,5)
error.precison_at_top_k(rating,5)
error.spearman_rank_corelation(rating,5)
end=time.time()
print("Time taken by the above model is "+str(end-start))

print("*****************************CUR WITH 90% retained Energy***************")
start=time.time()
cur=CUR()
#cur.create_matrices()
rating=cur.create_CUR(True)
#print(rating)
svd=SVD()
reating=svd.create_2d_rating(rating)
error=Error()
error.rmse(rating,6)
error.precison_at_top_k(rating,6)
error.spearman_rank_corelation(rating,6)
end=time.time()
print("Time taken by the above model is "+str(end-start))
'''