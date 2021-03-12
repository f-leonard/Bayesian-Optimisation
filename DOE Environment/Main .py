import numpy as np
import random
from numpy import array
from sklearn import metrics
import pandas
from Environment import environment
from Genethic_Algorithm import GA
from Graph_Comparison import graphComparison
from Graph_LinearRegression import graphLinearRegression
from Artificial_Neural_Network import get_model,model_fit


d = 0.1
a = 1
b = 1
class Agent(object):

    def agent(self):
        results=[]  #list of LSS
        data_train=[]  #list of parameters values
        data=[]
        pred_2=[]
        for cycle in range(10):   #10 first cycles to initially train the ANN
            a, b, c, d = self.random_parameters()
            a = (a - 1000) / 2500
            b = (b - 45) / 20
            c = (c - 2) / 2
            d = (d - 800) / 1760


            result=environment(c)  #LSS according to the environment
            data_train.append(a)   #the parameters are written in the list
            data_train.append(b)
            data_train.append(c)
            data_train.append(d)
            if result<0:
                results.append(0)
                pred_2.append(0)
            else:
                results.append(result[0][0])
                pred_2.append(result)

        data_train_2=data_train
        results_2=results
        data_train=np.reshape(data_train,(-1,4))  #the list is reshapped as a 4 column matrix
        results=np.reshape(results,(-1,1))  #the list is reshapped as a 1 column

        data=np.concatenate((data_train,results),axis=1)
        data=pandas.DataFrame(data=data)
        data.to_csv('data.csv')  #To save in the same file the parameters and the result
        data=pandas.read_csv('data.csv')
        data_train[0:,0]=data.iloc[0:,1].values
        data_train[0:,1]=data.iloc[0:,2].values
        data_train[0:,2]=data.iloc[0:,3].values
        data_train[0:,3]=data.iloc[0:,4].values
        results=data.iloc[0:,5].values
        best_weights=GA(data_train,results,8,12)    #GA calculates the weights
        best_weights[1]=np.squeeze(best_weights[1])
        best_weights[3]=np.squeeze(best_weights[3])
        model=get_model()  #Return the ANN model
        best_weights=model_fit(model,data_train,results)  #To train the ANN with the first cycles
        wrong_results=0
        good_results=0

        for i in range(10):  #number of different taken parameters
            a,b,c,d=self.random_parameters()
            a = (a - 1000) / 2500 #To normalize the 4 parameters


            d = (d - 800) / 1760
            data = array([[a,b,c,d]])

            result=model.predict(data)  #the ANN predicts the LSSS
            if result<0:
                pred_2.append(0)
            else:
                pred_2.append(result)
            env_result=environment(a,b,c,d)
            if env_result<0:
                results_2.append(0)
            else:
                results_2.append(env_result[0][0])
            data_train_2.append(a)
            data_train_2.append(b)
            data_train_2.append(c)
            data_train_2.append(d)
            data_train=np.reshape(data_train_2,(-1,4))
            results=np.reshape(results_2,(-1,1))
            pred=np.reshape(pred_2,(-1,1))
            data_train = pandas.DataFrame(data=data_train)
            results = pandas.DataFrame(data=results)
            data_train.to_csv('parameters.csv')
            results.to_csv('results.csv')
            pred = pandas.DataFrame(data=pred)
            pred.to_csv('predictions.csv')
            predictions=model.predict(data_train)  #The ANN makes predictions for all data
            error=metrics.mean_absolute_error(results,predictions)


            reward_2=abs((result-env_result)/result)
            if reward_2>0.05:
                wrong_results=wrong_results+1
                best_weights[1]=np.squeeze(best_weights[1])
                best_weights[3]=np.squeeze(best_weights[3])
                model=get_model()  #Return the ANN model
                best_weights=model_fit(model,data_train,results)

                print("number of wrong results= ",wrong_results," number of good results= ",good_results)
            else:
                good_results=good_results+1
                last_data=data
                parameter_choice=[0,1,2,3,4,5,6,7]  #to choose if increasing/decreasing a,b,c or d
                for i in range(8):
                    parameter=random.choice(parameter_choice)
                    parameter_choice.remove(parameter)
                    result_saved=model.predict(last_data)
                    data=last_data
                    e,f,g,h=a,b,c,d
                    reward_1=1

                    while (reward_1==1) and (e>=0) and (e<=1.2) and (f>=-0.25) and (f<=1) and (g>=0) and (g<=1) and (h>=0) and (h<=0.97):
                        result_saved=model.predict(data)
                        last_data=data
                        a,b,c,d=e,f,g,h
                        e,f,g,h=self.improved_LSS(a,b,c,d,parameter)
                        data=array([[e,f,g,h]])
                        result=model.predict(data)
                        print("result= ",result, " best result: ", result_saved, " a= ",e," b= ",f," c= ",g," d= ",h)
                        if result>result_saved:
                            reward_1=1
                        else:
                            reward_1=-1
                print("number of wrong results= ",wrong_results," number of good results= ",good_results)
            graphComparison(predictions,results,"Data Sample","Lap Shear Strength (N)","Comparison graph for final model")
            graphComparison(pred,results,"Data Sample","Lap Shear Strength (N)","Comparison graph for each iteration")
            graphLinearRegression(results,predictions,"Experimental Values (N)","Predicted Values (N)","Linear regression for final model",save=True)
            graphLinearRegression(results,pred,"Experimental Values (N)","Predicted Values (N)","Linear regression for each iteration")
            Final_weights=pandas.DataFrame(data=best_weights)
            Final_weights.to_csv('final_weights.csv',save=True)  #To store the weights
#This is the funciton that neesds to be replaced by bayesian optimisation.
    def improved_LSS(self,a,b,c,d,parameter):   #to increase or decrease a parameter chosen randomly

        if parameter==0:
            a=a+0.01
            return a,b,c,d
        if parameter==1:
            a=a-0.01
            return a,b,c,d
        if parameter==2:
            b=b+0.01
            return a,b,c,d
        if parameter==3:
            b=b-0.01
            return a,b,c,d
        if parameter==4:
            c=c+0.01
            return a,b,c,d
        if parameter==5:
            c=c-0.01
            return a,b,c,d
        if parameter==6:
            d=d+0.01
            return a,b,c,d
        if parameter==7:
            d=d-0.01
            return a,b,c,d
#this will also be replaced with the bayesian optimsation.
    def random_parameters(self):  #picking 4 random parameters in their normalized range
        a=random.randint(1000,4000)  #Choose a random welding energy value
        b=random.randint(40,65)   #Choose a random vibration amplitude value
        c=random.randint(2,4)   #Choose a random clamping pressure value
        d=random.randint(800,2500)   #Choose a random peak power value
        return a,b,c,d



agent=Agent()
agent.agent()

