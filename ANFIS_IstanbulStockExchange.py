
# coding: utf-8

# In[12]:


#Common Library
import numpy as np
import pandas as pd
from __future__ import division
from datetime import datetime,timedelta
from dateutil import parser
#import time
from time import time
#Sklearn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
#regression evaluation
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#Matplotlib
import matplotlib.pyplot as plt
# Anfis
#import anfis
#from membership import membershipfunction, mfDerivs


# In[13]:


def totimestamp(dt, epoch=datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 

def DataPrepare(df):
    df['TL BASED'][0] = 'ISE.TL'
    df['USD BASED'][0] = 'ISE.USD'
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace=True)
    df.rename(columns={'ISE': 'ISE.TL', 'ISE': 'ISE.USD'}, inplace=True)

    #col_target = ['ISE.TL', 'ISE.USD']
    col_target = ['ISE.TL']
    target = df[col_target]
    data = df.drop(['ISE.TL', 'ISE.USD'], axis=1)

    data['date'] = pd.to_datetime(data['date'])
    #print(data['date'][0])
    ms = []
    for i in range(1, len(data['date'])+1):
        #print(int(totimestamp(data['date'][i])))
        ms.append(int(totimestamp(data['date'][i])))
        
    data['date'] = ms

    return data, target


# In[15]:


def ReadData():
    df = pd.read_csv('IstanbulStockExchangeUCI/IstanbulStockExchangeUCI.csv', delimiter=';')
    #print(df.head())
    return DataPrepare(df)

#data, target = ReadData()


# In[16]:


def CreateFeatureSelectionModel(x_train, y_train, threshold = 0.1):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    sfm_m = RandomForestRegressor(max_depth=30, random_state=0)
    sfm = SelectFromModel(sfm_m, threshold=threshold)
    sfm.fit(x_train, y_train)
    return sfm

def DoFeatureSelection(model, data):
    return model.transform(data)

def DoInverseFeatureSelection(mode, data):
    return model.inverse_transform(data)


# In[17]:


def CreateDataScaleModel(x_train, y_train):
    from sklearn.preprocessing import MinMaxScaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    return scaler_x, scaler_y
    
    
def DataScale(model, data):
    return model.transform(data)


# In[21]:


def GetData(DoFS = True):
    data, target = ReadData()
    
    sfm = CreateFeatureSelectionModel(data, target, 0.25)
    if DoFS == True:
        X = DoFeatureSelection(sfm, data)
    else:
        X = data.values
    Y = target.values[:,0]
    aa = np.hstack([X,target.values])
    df = pd.DataFrame(data=aa)
    #print(df.head())
    df.to_csv('reduced.csv')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    #scale_x, scale_y = CreateDataScaleModel(x_train, y_train)
    #x_train = DataScale(scale_x, x_train)
    #y_train = DataScale(scale_y, y_train)
    return x_train, x_test, y_train, y_test

#aaa = GetData()


# In[205]:


def ConstructFISMembership(X, Y, sigma_lb, sigma_ub):
    mf = []

    for i in range(0, len(X[0])):
        _min = np.min(X[:,i])
        _max = np.max(X[:,i])
        _mean = np.mean(X[:,i])
        vmf = []
        vmf.append(['gaussmf',{'mean':_min,'sigma':random.uniform(sigma_lb, sigma_ub)}])
        vmf.append(['gaussmf',{'mean':random.uniform(_min, _max),'sigma':random.uniform(sigma_lb, sigma_ub)}])
        vmf.append(['gaussmf',{'mean':_max,'sigma':random.uniform(sigma_lb, sigma_ub)}])
        mf.append(vmf)

    print(mf)
    mfc = anfis.membership.membershipfunction.MemFuncs(mf)
    return mfc


# In[206]:


def CreateANFISModel(X, Y, sigma_lb=1.25, sigma_ub=5, epoch=30):
    start=datetime.now()
    mfc = ConstructFISMembership(X, Y, sigma_lb=sigma_lb, sigma_ub=sigma_ub)
    anf = anfis.anfis.ANFIS(X, Y, mfc)
    anf.trainHybridJangOffLine(epochs=epoch)
    Y_ = anfis.anfis.predict(anf, X)[:,0]
    print('train var : ' + str(explained_variance_score(Y, Y_)))
    print('train r2 : ' + str(r2_score(Y, Y_)))
    print('train rmse : ' + str(np.sqrt(mean_squared_error(Y, Y_))))
    print(datetime.now()-start)
    return anf


# In[207]:


def ANFISEvaluate(model, X, Y):
    y_pred = anfis.anfis.predict(model, X)[:,0]
    y_true = Y
    print(y_pred[0:10])
    print(y_true[0:10])
    #y_pred = scaler_y.inverse_transform(y_pred)
    #y_test = scaler_y.inverse_transform(y_test)
    print('test var : ' + str(explained_variance_score(y_true, y_pred)))
    print('test r2 : ' + str(r2_score(y_true, y_pred)))
    print('test rmse : ' + str(np.sqrt(mean_squared_error(y_true, y_pred))))
    print round(anf.consequents[-1][0],6)
    print round(anf.consequents[-2][0],6)
    print round(anf.fittedValues[9][0],6)
    if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
        print 'test is good'
    anf.plotErrors()
    anf.plotResults()
    x_plot = range(0, len(X))
    plt.scatter(x_plot, y_pred, marker='^', color='blue')
    plt.scatter(x_plot, y_true, marker='o', color='red')
    plt.show()


# In[208]:


x_train, x_test, y_train, y_test = GetData()
anf = CreateANFISModel(x_train, y_train, sigma_lb=0.75, sigma_ub=5)
ANFISEvaluate(anf, x_test, y_test)


# In[5]:


def CreateSVRModel(X, Y, C=1.0, epsilon=0.01):
    if len(X) != len(Y):
        print('x and y not have same size')
        return
    #start=datetime.now()
    clf = SVR(kernel='rbf', degree=3, C=C, epsilon=epsilon)
    #clf = RandomForestRegressor(max_depth=20, random_state=0)
    clf.fit(X, Y)
    Y_ = clf.predict(X)
    #print('train var : ' + str(explained_variance_score(Y, Y_)))
    #print('train r2 : ' + str(r2_score(Y, Y_)))
    #print('train rmse : ' + str(np.sqrt(mean_squared_error(Y, Y_))))
    #print(datetime.now()-start)
    return clf

def SVMEvaluate(model, X, Y):
    if len(X) != len(Y):
        print('x and y not have same size')
        return
    y_pred = model.predict(X)
    y_true = Y
    print(y_pred[13:23])
    print(y_true[13:23])
    print('test var : ' + str(explained_variance_score(y_true, y_pred)))
    print('test r2 : ' + str(r2_score(y_true, y_pred)))
    print('test rmse : ' + str(np.sqrt(mean_squared_error(y_true, y_pred))))
    x_plot = range(0, len(Y))
    plt.scatter(x_plot, y_pred, marker='^', color='blue')
    plt.scatter(x_plot, y_true, marker='o', color='red')
    plt.show()


# In[40]:


x_train, x_test, y_train, y_test = GetData(False)
svr_model = CreateSVRModel(x_train, y_train)
SVMEvaluate(svr_model, x_test, y_test)


# In[215]:


import math
from optimal import GenAlg
from optimal import Problem
from optimal import helpers

SVM_ACO_Points = []

def SVMGAEvaluate(model, X, Y):
    if len(X) != len(Y):
        print('x and y not have same size')
        return
    y_pred = model.predict(X)
    y_true = Y
    var = explained_variance_score(y_true, y_pred)
    print('test var : ' + str(var))
    print('test r2 : ' + str(r2_score(y_true, y_pred)))
    print('test rmse : ' + str(np.sqrt(mean_squared_error(y_true, y_pred))))
    print('\n')
    return var
    #x_plot = range(0, len(X))
    #plt.scatter(x_plot, y_pred, marker='^', color='blue')
    #plt.scatter(x_plot, y_true, marker='o', color='red')
    #plt.show()

def decode_svmga_search_space(binary):
    c = helpers.binary_to_float(binary[0:16], 1, 20.0)
    epsilon = helpers.binary_to_float(binary[16:32], 0.001, 0.01)
    return c, epsilon

def svmga_fitness(solution):
    c, epsilon = solution
    
    x_train, x_test, y_train, y_test = GetData()
    
    svr_model = CreateSVRModel(x_train, y_train, C=c, epsilon=epsilon)
    SVM_ACO_Points.append((c, epsilon))
    output = SVMGAEvaluate(svr_model, x_test, y_test)
    finished = output
    fitness = output
    return fitness, finished

def SVMGA():
    svmga = Problem(svmga_fitness, decode_function=decode_svmga_search_space)
    my_genalg = GenAlg(32)
    best_solution = my_genalg.optimize(svmga, max_iterations=1000000)
    print best_solution


# In[4]:


from optimal import GenAlg
from optimal import Problem
from optimal import helpers


# In[216]:


start=datetime.now()
SVMGA()
print(datetime.now()-start)
SVM_ACO_Points


# In[1019]:


x_train, x_test, y_train, y_test = GetData()
svr_model = CreateSVRModel(x_train, y_train, C=19.989852750438697, epsilon=0.024782330052643627)
SVMEvaluate(svr_model, x_test, y_test)


# In[211]:


import math
from optimal import GenAlg
from optimal import Problem
from optimal import helpers

def ANFISGAEvaluate(model, X, Y):
    if len(X) != len(Y):
        print('x and y not have same size')
        return
    y_pred = anfis.anfis.predict(model, X)
    y_true = Y
    var = explained_variance_score(y_true, y_pred)
    print('test var : ' + str(var))
    print('test r2 : ' + str(r2_score(y_true, y_pred)))
    print('test rmse : ' + str(np.sqrt(mean_squared_error(y_true, y_pred))))
    print('\n')
    anf.plotErrors()
    anf.plotResults()
    x_plot = range(0, len(X))
    plt.scatter(x_plot, y_pred, marker='^', color='blue')
    plt.scatter(x_plot, y_true, marker='o', color='red')
    plt.show()
    return var

def decode_anfisga_search_space(binary):
    sigma_lb = helpers.binary_to_float(binary[0:16], 0.25, 5)
    sigma_ub = helpers.binary_to_float(binary[16:32], 5, 10)
    return sigma_lb, sigma_ub

def anfisga_fitness(solution):
    sigma_lb, sigma_ub = solution
    
    x_train, x_test, y_train, y_test = GetData()
    anf = CreateANFISModel(x_train, y_train, sigma_lb=sigma_lb, sigma_ub=sigma_ub)
    output = ANFISGAEvaluate(anf, x_test, y_test)
    finished = output
    fitness = output
    return fitness, finished

def ANFISGA():
    anfisga = Problem(anfisga_fitness, decode_function=decode_anfisga_search_space)
    my_genalg = GenAlg(32)
    best_solution = my_genalg.optimize(anfisga)
    print best_solution


# In[212]:


start=datetime.now()
ANFISGA()
print(datetime.now()-start)


# In[28]:


'''
    ==============================================================
    Ant Colony Optimization algorithm for continuous domains ACO_R
    ==============================================================
    author: Andreas Tsichritzis <tsadreas@gmail.com>
'''

import os
import sys
import shutil
import math

import multiprocessing
#import datetime

from scipy.stats import norm

from collections import defaultdict
from operator import itemgetter




def svm_evaluator(x):
    '''Evaluator function, returns fitness and responses values'''
    # give the normalized candidates values inside the real design space
    #x= [10*i-5 for i in x]
    #print(x)
    #f = (sum([math.pow(i,4)-16*math.pow(i,2)+5*i for i in x])/2)
    
    x_train, x_test, y_train, y_test = GetData()
    svr_model = CreateSVRModel(x_train, y_train, C=x[0], epsilon=x[1])
    y_pred = svr_model.predict(x_test)
    y_true = y_test
    var = explained_variance_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    f = var
    print('var : ' + str(var))
    print('r2 : ' + str(r2))
    print('rmse : ' + str(rmse))
    print('c : ' + str(x[0]))
    print('epsilon : ' + str(x[1]))
    # calculate values for other responses
    res = {'r1':f-5,'r2':2*f}
    fitness = dict(Obj=f,**res)
    return fitness

def anfis_evaluator(x):
    '''Evaluator function, returns fitness and responses values'''
    # give the normalized candidates values inside the real design space
    #x= [10*i-5 for i in x]
    #print(x)
    #f = (sum([math.pow(i,4)-16*math.pow(i,2)+5*i for i in x])/2)

    x_train, x_test, y_train, y_test = GetData()
    anf = CreateANFISModel(x_train, y_train, sigma_lb=x[0], sigma_ub=x[1])
    y_pred = anfis.anfis.predict(anf, x_test)
    y_true = y_test
    var = explained_variance_score(y_true, y_pred)
    f = var
    #print('f : ' + str(f))
    # calculate values for other responses
    res = {'r1':f-5,'r2':2*f}
    fitness = dict(Obj=f,**res)
    return fitness

def mp_evaluator(x, func):
    '''Multiprocessing evaluation'''
    # ste number of cpus
    
    nprocs = 2
    # create pool
    pool = multiprocessing.Pool(processes=nprocs)
    results = [pool.apply_async(func,[c]) for c in x]
    #print(results)
    pool.close()
    pool.join()
    f = [r.get()['Obj'] for r in results]
    for r in results:
        del r.get()['Obj']
    # maximization or minimization problem
    
    maximize = False
    res = (f, [r.get() for r in results], maximize)
    #print(len(res))
    #print(np.array(res))
    return res


def initialize(ants,var,ul_bound):
    '''Create initial solution matrix'''   
    X = []
    for i in range(0, var):
        X.append(np.random.uniform(ul_bound[i][0], ul_bound[i][1], size=(ants, 1)))
    #X = np.random.uniform(low=0,high=1, size=(ants,var))
    return np.hstack(X)


def init_observer(filename,matrix,parameters,responses):
    '''Initial population observer'''
    p = []
    r = []
    f = []
    res = ['{0:>10}'.format(i)[:10] for i in responses]
    par = ['{0:>10}'.format(i)[:10] for i in parameters]
    for i in range(len(matrix)):
        p.append(matrix[i][0:len(parameters)])
        r.append(matrix[i][len(parameters):-1])
        f.append(matrix[i][-1])
    r = np.array(r)
    p = np.array(p)

    for i in range(len(r)):
        r[i] = ['{0:>10}'.format(r[i][j])[:10] for j in range(len(responses))]

    for i in range(len(p)):
        p[i] = ['{0:>10}'.format(p[i][j])[:10] for j in range(len(parameters))]

    f = ['{0:>10}'.format(i)[:10] for i in f]

    iteration = 0

    filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format('Iteration',', '.join(map(str, par)),'Fitness',', '.join(map(str, res))))

    for i in range(len(matrix)):
        filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format(iteration,', '.join(map(str, p[i])),f[i],', '.join(map(str, r[i]))))



def iter_observer(filename,matrix,parameters,responses,iteration):
    '''Iterations observer'''
    p = []
    r = []
    f = []
    for i in range(len(matrix)):
        p.append(matrix[i][0:len(parameters)])
        r.append(matrix[i][len(parameters):-1])
        f.append(matrix[i][-1])
    r = np.array(r)
    p = np.array(p)

    for i in range(len(r)):
        r[i] = ['{0:>10}'.format(r[i][j])[:10] for j in range(len(responses))]

    for i in range(len(p)):
        p[i] = ['{0:>10}'.format(p[i][j])[:10] for j in range(len(parameters))]

    f = ['{0:>10}'.format(i)[:10] for i in f]

    for i in range(len(matrix)):
        filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format(iteration,', '.join(map(str, p[i])),f[i],', '.join(map(str, r[i]))))




def formatTD(td):
    """ Format time output for report"""
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return '%s days %s h %s m %s s' % (days, hours, minutes, seconds)

def evolve(display, func, var_names, ul_bound, nAnts = 8, pheromone_evaporation = 0.65, max_iterations = 300):
    '''Executes the optimization'''

    #start_time = time()
    start_time = datetime.now()

    # number of variables
    parameters_v = var_names
    response_v = ['r1','r2']

    # create output file
    projdir = os.getcwd()
    ind_file_name = '{0}/results.csv'.format(projdir)
    ind_file = open(ind_file_name, 'w')

    # number of variables
    nVar = len(parameters_v)
    # size of solution archive
    nSize = 8
    # number of ants
    nAnts = nAnts

    # parameter q
    q = 0.3

    # standard deviation
    qk = q*nSize

    # parameter xi (like pheromone evaporation)
    xi = pheromone_evaporation

    # maximum iterations
    maxiter = max_iterations
    # tolerance
    errormin = 0.01

    # bounds of variables
    #Up = [1]*nVar
    #Lo = [0]*nVar
    Up = [np.max(ul_bound)]*nVar
    Lo = [np.min(ul_bound)]*nVar

    # initilize matrices
    S = np.zeros((nSize,nVar))
    S_f = np.zeros((nSize,1))

    plt.figure()


    # initialize the solution table with uniform random distribution and sort it
    print('-----------------------------------------')
    print('Starting initilization of solution matrix')
    print('-----------------------------------------')
    
    
    Srand = initialize(nSize, nVar, ul_bound)
    #print(Srand)
    f,S_r,maximize = mp_evaluator(Srand, func)

    S_responses = []

    for i in range(len(S_r)):
        S_f[i] = f[i]
        k = S_r[i]
        row = []
        for r in response_v:
            row.append(k[r])
        S_responses.append(row)

    # add responses and "fitness" column to solution
    S = np.hstack((Srand,S_responses,S_f))
    # sort according to fitness (last column)
    S = sorted(S, key=lambda row: row[-1],reverse = maximize)
    S = np.array(S)

    init_observer(ind_file,S,parameters_v,response_v)

    # initilize weight array with pdf function
    w = np.zeros((nSize))
    for i in range(nSize):
        w[i] = 1/(qk*2*math.pi)*math.exp(-math.pow(i,2)/(2*math.pow(q,2)*math.pow(nSize,2)))


    if display:
        x = []
        y = []
        for i in S:
            x.append(i[0])
            y.append(i[1])

        plt.scatter(x,y)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.pause(2)
        plt.cla()

    # initialize variables
    iterations = 1
    best_par = []
    best_obj = []
    best_sol = []
    best_res = []
    worst_obj = []
    best_par.append(S[0][:nVar])
    best_obj.append(S[0][-1])
    best_sol.append(S[0][:])
    best_res.append(S[0][nVar:-1])
    worst_obj.append(S[-1][-1])

    stop = 0

    # iterations
    while True:
        #print '-----------------------------------------'
        print('Iteration', iterations)
        #print '-----------------------------------------'
        # choose Gaussian function to compose Gaussian kernel
        p = w/sum(w)

        # find best and index of best
        max_prospect = np.amax(p)
        ix_prospect = np.argmax(p)
        selection = ix_prospect

        # calculation of G_i
        # find standard deviation sigma
        sigma_s = np.zeros((nVar,1))
        sigma = np.zeros((nVar,1))
        for i in range(nVar):
            for j in range(nSize):
                sigma_s[i] = sigma_s[i] + abs(S[j][i] - S[selection][i])
            sigma[i] = xi / (nSize -1) * sigma_s[i]


        Stemp = np.zeros((nAnts,nVar))
        ffeval = np.zeros((nAnts,1))
        res = np.zeros((nAnts,len(response_v)))
        for k in range(nAnts):
            for i in range(nVar):
                Stemp[k][i] = sigma[i] * np.random.random_sample() + S[selection][i]
                if Stemp[k][i] > Up[i]:
                    Stemp[k][i] = Up[i]
                elif Stemp[k][i] < Lo[i]:
                    Stemp[k][i] = Lo[i]
        f,S_r,maximize = mp_evaluator(Stemp, func)

        S_f = np.zeros((nAnts,1))
        S_responses = []

        for i in range(len(S_r)):
            S_f[i] = f[i]
            k = S_r[i]
            row = []
            for r in response_v:
                row.append(k[r])
            S_responses.append(row)

        # add responses and "fitness" column to solution
        Ssample = np.hstack((Stemp,S_responses,S_f))

        # add new solutions in the solutions table
        Solution_temp = np.vstack((S,Ssample))

        # sort according to "fitness"
        Solution_temp = sorted(Solution_temp, key=lambda row: row[-1],reverse = maximize)
        Solution_temp = np.array(Solution_temp)

        # keep best solutions
        S = Solution_temp[:nSize][:]

        # keep best after each iteration
        best_par.append(S[0][:nVar])
        best_obj.append(S[0][-1])
        best_res.append(S[0][nVar:-1])
        best_sol.append(S[0][:])
        worst_obj.append(S[-1][-1])

        iter_observer(ind_file,S,parameters_v,response_v,iterations)

        if display:
            # plot new table
            x = []
            y = []
            for i in S:
                x.append(i[0])
                y.append(i[1])

            plt.scatter(x,y)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.pause(2)

        if iterations > 1:
            diff = abs(best_obj[iterations]-best_obj[iterations-1])
            if diff <= errormin:
                stop += 1

        iterations += 1
        if iterations > maxiter or stop > 5:
            break
        else:
            if display:
                plt.cla()

    ind_file.close()

    total_time_s = datetime.now() - start_time
    
    #total_time = timedelta(seconds=total_time_s)
    #total_time = formatTD(total_time)

    # fix varibales values in output file
    #correct_par(ind_file_name,parameters_v)

    best_sol = sorted(best_sol, key=lambda row: row[-1],reverse = maximize)

    print("Best individual:", parameters_v)
    print(best_sol[0][0:len(parameters_v)])
    print("Fitness:")
    print(best_sol[0][-1])
    print("Responses:", response_v)
    print(best_sol[0][len(parameters_v):-1])


# Executes optimization run.
# If display = True plots ants in 2D design space
#evolve(display = False)


# In[29]:


start=datetime.now()
ul_bound = [[1, 20], [0.01, 0.001]]
parameter_names = ['c', 'epsilon']
evolve(False, svm_evaluator, parameter_names, ul_bound, nAnts = 50)
print(datetime.now()-start)


# In[374]:


start=datetime.now()
ul_bound = [[0.25, 5], [5, 10]]
parameter_names = ['sigma_ub', 'sigma_lb']
evolve(False, anfis_evaluator, parameter_names, ul_bound, nAnts = 50)
print(datetime.now()-start)

