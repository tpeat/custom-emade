import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, func
import math

# from selection_methods
def find_pareto(data):
    is_Pareto = np.ones(data.shape[0], dtype = bool)
    for i, c in enumerate(data):
        # Keep any point with a lower cost
        if is_Pareto[i]:
            # This is where you would change for miniminzation versus maximization 
            # Minimization
            is_Pareto[is_Pareto] = np.any(data[is_Pareto]<c, axis=1)  
            # Maximization
            #is_Pareto[is_Pareto] = np.any(data[is_Pareto]>c, axis=1)  
            # And keep self
            is_Pareto[i] = True  
    # Downsample from boolean array
    Pareto_data = data[is_Pareto, :]
    # Sort data
    Pareto_out =  Pareto_data[np.argsort(Pareto_data[:,0])]
    #return is_Pareto
    return Pareto_out, is_Pareto

# the call for localhost might be: sql_connect('root', 'localhost', '', 'titanic')
def sql_connect(username, ip, password, db):
    engine = create_engine('mysql+pymysql://' + username + password + ':@' + ip + ':3306/' + db)
    con = engine.connect()
    return con

def plot_auc(myPareto):
    # Calculate the Area under the Curve as a Riemann sum
    auc = np.sum(np.diff(myPareto[:,0])*myPareto[0:-1,1])
    # Create figure
    plt.figure()
    # Make sure font sizes are large enough to read in the presentation
    plt.rcParams.update({'font.size': 14})
    # Plot Pareto steps. note 'post' for minimization 'pre' for maximization
    plt.step(myPareto[:,0], myPareto[:,1], where='post')
    #plt.step(myPareto[:,0], myPareto[:,1], where='pre')
    # Make sure you include labels
    # Minimization
    plt.title('Example of a Minimization Result\n with AUC = ' + str(auc))
    plt.xlabel('False Negative Rate')
    plt.ylabel('False Positive Rate')
    plt.savefig('paretoFrontEMADE.png', bbox_inches='tight')
    plt.show()

# get connection
connection = sql_connect('root', 'localhost', '', 'titanic')

# query
data = connection.execute('Select * from individuals join paretofront on individuals.hash=paretofront.hash where paretofront.generation=(select max(generation) from paretofront)')
paretoFrontTable = []
#Only required if trying to find the trees of elements in paretoFront
paretoFrontTreeTable = []
for row in data:
    paretoFrontTable.append([row['FullDataSet False Negative Rate'], row['FullDataSet False Positive Rate']])
    paretoFrontTreeTable.append([row['FullDataSet False Negative Rate'], row['FullDataSet False Positive Rate'], row['tree']])

# Use above routine to find pareto points
#paretoFrontTable = np.vstack(([[0,1],[1,0]], paretoFrontTable))
paretoFrontTable = np.vstack(([[0,1],[1,0]], paretoFrontTable))
myPareto, isPareto = find_pareto(np.array(paretoFrontTable))
#myPareto = myPareto[np.argsort(myPareto[:,0])]

#Figures out what tree produced the min-distance point on pareto-front
myParetoList = myPareto.tolist()
min_entry = min(myParetoList, key=lambda x: math.sqrt(x[0]**2 + x[1]**2))
for entry in paretoFrontTreeTable:
    if entry[0] == min_entry[0] and entry[1] == min_entry[1]:
        print(entry[0], entry[1], entry[2])

plot_auc(myPareto)