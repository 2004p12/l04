week1

graph = {}
edge_set = set()
# Add a node only if it doesn't already exist
def add_node(node):
    if node in graph:
        print(f"'{node}' already exists. Please enter a different node.")
        return False
    graph[node] = []
    return True
# Add edge only if not a duplicate
def add_edge(u, v):
    edge = tuple(sorted((u, v)))
    if edge in edge_set:
        print(f"Edge {u}-{v} already exists. Please enter a different edge.")
        return False
    if u not in graph or v not in graph:
        print("Both nodes must be added before connecting them with an edge.")
        return False

    graph[u].append(v)
    graph[v].append(u)
    edge_set.add(edge)
    return True
# BFS
def bfs(start):
    visited = []
    queue = [start]
    print("BFS:", end=" ")

    while queue:
        node = queue.pop(0)
        if node not in visited:
            print(node, end=" ")
            visited.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    print()
# DFS
def dfs(node, visited=None):
    if visited is None:
        visited = []
        print("DFS:", end=" ")

    if node not in visited:
        print(node, end=" ")
        visited.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, visited)
# === Input Section ===
# Add unique nodes
n = int(input("Enter number of nodes: "))
i = 0
while i < n:
    node = input(f"Enter node {i + 1}: ").strip()
    if add_node(node):
        i += 1

# Add edges without duplication
e = int(input("Enter number of edges: "))
for i in range(e):
    while True:
        u, v = input(f"Enter edge {i + 1} (two nodes): ").split()
        if add_edge(u, v):
            break
# Start traversal
start = input("Enter starting node: ").strip()
if start in graph:
    bfs(start)
    dfs(start)
    print()
else:
    print("Starting node not found in the graph.")


week2


def aStarAlgo(start_node, stop_node):
    open_set = {start_node}  # Set of nodes to be evaluated
    closed_set = set()  # Set of nodes already evaluated
    g = {}  # Dictionary to store the distance from the start node
    parents = {}  # Dictionary to store the parent of each node

    # The distance from the start node to itself is zero
    g[start_node] = 0
    # The start node has no parent (it is the root)
    parents[start_node] = start_node

    while open_set:
        n = None

        # Node with the lowest f() = g + heuristic() value is chosen
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        # Print current node being evaluated, heuristic value and the sets
        print(f"\nEvaluating node: {n} (g: {g[n]}, h: {heuristic(n)}, f: {g[n] + heuristic(n)})")
        print(f"Open Set: {open_set}")
        print(f"Closed Set: {closed_set}")

        # If the goal is reached or no more nodes can be explored
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()  # Reverse the path to get from start to goal
            print('Path found: {}'.format(path))
            return path

        # Explore neighbors of the current node
        print(f"Exploring neighbors of {n}:")
        for (m, weight) in get_neighbors(n):
            h_m = heuristic(m)  # Heuristic value of the neighbor
            print(f"  Neighbor: {m} with weight: {weight} and h({m}): {h_m}")

            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight
                print(f"  Added {m} to open set with g({m}) = {g[m]} and f({m}) = {g[m] + h_m}")
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)
                        print(f"  Updated {m} to have a shorter path with g({m}) = {g[m]} and f({m}) = {g[m] + h_m}")

        open_set.remove(n)
        closed_set.add(n)

    print('Path does not exist!')
    return None

# Function to return neighbors and their distances
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    return []

# Heuristic function for each node (Manhattan or other heuristic values)
def heuristic(n):
    H_dist = {
        'S': 5,
        'A': 3,
        'B': 4,
        'C': 2,
        'D': 6,
        'G': 0,
    }
    return H_dist.get(n, 0)

# Graph representation (node -> list of (neighbor, weight))
Graph_nodes = {
    'S': [('A', 1), ('G', 10)],
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 5)],
    'C': [('D', 3), ('G', 4)],
    'D': [('G', 2)],
}

# Run the algorithm
aStarAlgo('S', 'G')


week3

           from sys import maxsize  # Import maxsize to represent an infinitely large value (for comparison)
from itertools import permutations  # Import permutations to generate all possible orders of cities

v = 4  # Number of cities (vertices), here v = 4 (cities labeled 0, 1, 2, 3)

# Function to find the shortest path for the Traveling Salesman Problem
def travellingSalesmanProblem(graph, s):
    vertex = []  # List to store all cities excluding the start city 's'

    # Loop through all the cities and add those that are not the starting city 's'
    for i in range(v):
        if i != s:
            vertex.append(i)  # Add city i to the list 'vertex' if it's not the start city

    # Initialize min_path to a very large value (infinity), so it can be updated with valid paths
    min_path = maxsize

    # Generate all possible permutations of the cities in 'vertex'
    # This will give all possible orders of visiting the cities
    next_permutation = permutations(vertex)

    # Iterate through each permutation of cities
    for i in next_permutation:
        current_pathweight = 0  # Initialize the total distance of the current path
        k = s  # Start at the initial city (starting city is 's')

        # Loop through the cities in the current permutation
        for j in i:
            current_pathweight += graph[k][j]  # Add the distance from the current city 'k' to the next city 'j'
            k = j  # Update the current city to the next city

        # After visiting all cities, add the distance to return to the starting city
        current_pathweight += graph[k][s]  # Add the distance from the last city back to the start city

        # Update min_path if the current path weight is smaller than the previous min_path
        min_path = min(min_path, current_pathweight)

    # Return the shortest path found after evaluating all permutations
    return min_path

# Example graph where the value graph[i][j] represents the distance from city 'i' to city 'j'
graph = [
    [0, 10, 15, 20],  # Distances from city 0 to others
    [10, 0, 35, 25],  # Distances from city 1 to others
    [15, 35, 0, 30],  # Distances from city 2 to others
    [20, 25, 30, 0]   # Distances from city 3 to others
]

s = 0  # Set the starting city (index 0, which is city 0)

# Call the function and print the result
print(travellingSalesmanProblem(graph, s))  #

colors=['Red','Blue','Green']
states=['a','b','c','d']
neighbors={}
neighbors['a']=['b','c','d']
neighbors['b']=['a','d']
neighbors['c']=['a','d']
neighbors['d']=['c','b','a']

colors_of_states={}

def promising(state,color):#d,green
    for neighbor in neighbors.get(state):#c,b,a
        color_of_neighbor=colors_of_states.get(neighbor)#blue
        if color_of_neighbor==color:#b==b
            return False
    return True

def get_color_for_state(state):#d
    for color in colors:#Red,Blue,Green
        if promising(state,color):#d,Red
            return color

def main():
    for state in states:#c,d
        colors_of_states[state]=get_color_for_state(state)#a:Red,b:blue,c:blue,d:green

    print(colors_of_states)

main()

                        
 week-4
from sympy import symbols, Or, Not, Implies,Xor,satisfiable
 
Rain = symbols('Rain')
Harry_Visited_Hagrid = symbols('Harry_Visited_Hagrid')
Harry_Visited_Dumbledore = symbols('Harry_Visited_Dumbledore')

# Define the logical expressions based on the given statements
sentence_1 = Implies(Not(Rain), Harry_Visited_Hagrid)
sentence_2 = Xor(Harry_Visited_Hagrid, Harry_Visited_Dumbledore)
sentence_3 = Harry_Visited_Dumbledore

# Combine the statements
knowledge_base = sentence_1 & sentence_2 & sentence_3

#Finding the solution  
solution = satisfiable(knowledge_base, all_models=True)

#To print the output
for model in solution:
    if model[Rain]:
        print("It rained today.")
    else:
        print("There is no rain today.")





                 #week5

#Bayesian Network
# Define conditional probability tables (CPTs)
P_burglary = 0.002#t
P_earthquake = 0.001#t

# Probability of alarm given burglary and earthquake
P_alarm_given_burglary_and_earthquake = 0.94
P_alarm_given_burglary_and_no_earthquake = 0.95
P_alarm_given_no_burglary_and_earthquake = 0.31
P_alarm_given_no_burglary_and_no_earthquake = 0.001

# Probability of David calling given alarm
P_david_calls_given_alarm = 0.91#t
P_david_does_not_call_given_alarm = 0.09
P_david_calls_given_no_alarm = 0.05#t
P_david_does_not_call_given_no_alarm = 0.95

# Probability of Sophia calling given alarm
P_sophia_calls_given_alarm = 0.75
P_sophia_does_not_call_given_alarm = 0.25
P_sophia_calls_given_no_alarm = 0.02
P_sophia_does_not_call_given_no_alarm = 0.98

# Calculate joint probability
def joint_probability(alarm, burglary, earthquake, david_calls, sophia_calls):#(t,f,f,t,t)
    if alarm:
        if burglary and earthquake:
            P_alarm = P_alarm_given_burglary_and_earthquake
        elif burglary:
            P_alarm = P_alarm_given_burglary_and_no_earthquake
        elif earthquake:
            P_alarm = P_alarm_given_no_burglary_and_earthquake
        else:
            P_alarm = P_alarm_given_no_burglary_and_no_earthquake#0.001
    else:
        if burglary and earthquake:
            P_alarm = 1 - P_alarm_given_burglary_and_earthquake
        elif burglary:
            P_alarm = 1 - P_alarm_given_burglary_and_no_earthquake
        elif earthquake:
            P_alarm = 1 - P_alarm_given_no_burglary_and_earthquake
        else:
            P_alarm = 1 - P_alarm_given_no_burglary_and_no_earthquake

    P_david = (P_david_calls_given_alarm if david_calls else P_david_does_not_call_given_alarm) if alarm else (P_david_calls_given_no_alarm if david_calls else P_david_does_not_call_given_no_alarm)#0.91

    P_sophia = (P_sophia_calls_given_alarm if sophia_calls else P_sophia_does_not_call_given_alarm) if alarm else (P_sophia_calls_given_no_alarm if sophia_calls else P_sophia_does_not_call_given_no_alarm)#0.75

    return (P_burglary if burglary else 1 - P_burglary) * (P_earthquake if earthquake else 1 - P_earthquake) * P_alarm * P_david * P_sophia#0.75*0.91*0.001*0.998*0.999

# Calculate the probability for the given scenario
result = joint_probability(
    alarm=True,
    burglary=False,
    earthquake=False,
    david_calls=True,
    sophia_calls=True
)


# Print the result
print(f'The probability that the alarm has sounded, there is neither a burglary nor an earthquake, and both David and Sophia called Harry is: {result:.8f}')



week6

import numpy as np
import itertools
import pandas as pd

# Define state space and probabilities
states = ['sleeping', 'eating', 'pooping']
hidden_states = ['healthy', 'sick']
pi = [0.5, 0.5]  # Initial state probabilities

# Initial state distribution
state_space = pd.Series(pi, index=hidden_states, name='states')
print("Initial Probabilities:\n", state_space, "\n")

# Transition probabilities (hidden -> hidden)
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc['healthy'] = [0.7, 0.3]
a_df.loc['sick'] = [0.4, 0.6]
print("Transition Probabilities:\n", a_df, "\n")

# Emission probabilities (hidden -> observable)
b_df = pd.DataFrame(columns=states, index=hidden_states)
b_df.loc['healthy'] = [0.2, 0.6, 0.2]
b_df.loc['sick'] = [0.4, 0.1, 0.5]
print("Emission Probabilities:\n", b_df, "\n")

# Forward algorithm: Total probability of observation sequence
def forward_algorithm(obs_seq, a_df, b_df, pi, hidden_states):
    total_prob = 0
    all_state_paths = list(itertools.product(hidden_states, repeat=len(obs_seq)))

    for path in all_state_paths:
        prob = pi[hidden_states.index(path[0])] * b_df.loc[path[0], obs_seq[0]]
        for t in range(1, len(obs_seq)):
            prev_state = path[t - 1]
            curr_state = path[t]
            prob *= a_df.loc[prev_state, curr_state] * b_df.loc[curr_state, obs_seq[t]]
        total_prob += prob

    return total_prob

# Viterbi algorithm: Most likely hidden state sequence
def viterbi_algorithm(obs_seq, a_df, b_df, pi, hidden_states):
    max_prob = 0
    best_path = None
    all_state_paths = list(itertools.product(hidden_states, repeat=len(obs_seq)))

    for path in all_state_paths:
        prob = pi[hidden_states.index(path[0])] * b_df.loc[path[0], obs_seq[0]]
        for t in range(1, len(obs_seq)):
            prev_state = path[t - 1]
            curr_state = path[t]
            prob *= a_df.loc[prev_state, curr_state] * b_df.loc[curr_state, obs_seq[t]]
        if prob > max_prob:
            max_prob = prob
            best_path = path

    return max_prob, best_path

# Example observation sequence
obsq = ['sleeping', 'eating', 'sleeping']

# Run and print
print("Forward (total probability):", forward_algorithm(obsq, a_df, b_df, pi, hidden_states))
v_prob, v_path = viterbi_algorithm(obsq, a_df, b_df, pi, hidden_states)
print("Viterbi (most probable state path):", v_path)
print("Viterbi probability:", v_prob)


week7

implement Regression algorithm
Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/content/Salary_Data.csv')
print(dataset)

X = dataset.iloc[:, :-1].values  #independent variable array
y = dataset.iloc[:,1].values  #dependent variable vector

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)


# fitting the regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

predicting the test set results
y_pred = regressor.predict(X_test)
print(X_test)
y_test

# visualizing the results
#plot for the TRAIN
plt.scatter(X_train, y_train, color='red') # plotting the observation line
plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line
plt.title("Salary vs Experience (Training set)") # stating the title of the graph
plt.xlabel("Years of experience") # adding the name of x-axis
plt.ylabel("Salaries") # adding the name of y-axis
plt.show() # specifies end of graph


#plot for the TEST
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line
plt.title("Salary vs Experience (Testing set)")# stating the title of the graph
plt.xlabel("Years of experience")# adding the name of x-axis
plt.ylabel("Salaries")# adding the name of y-axis
plt.show()# specifies end of graph


week 9

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
iris =datasets.load_iris()
X=pd.DataFrame(iris.data)
X.columns=['Sepal_Length','Sepal_Width', 'Petal_length', 'Petal_Width']
print(X)
y=pd.DataFrame(iris.target)
y.columns=['target']
print(y)

plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])
plt.subplot(1,2,1)
plt.scatter(X.Sepal_Length,X.Sepal_Width,c=colormap[y.target],s=40)
plt.title('Sepal')
plt.subplot(1,2,2)
plt.scatter(X.Petal_length,X.Petal_Width,c=colormap[y.target],s=40)
plt.title('Petal')
plt.show()

model=KMeans(n_clusters=3)
model.fit(X)
print(model.labels_)
plt.subplot(1,2,1)
plt.scatter(X.Petal_length,X.Petal_Width,c=colormap[y.target],s=40)
plt.title('Real Classification')
plt.subplot(1,2,2)
plt.scatter(X.Petal_length,X.Petal_Width,c=colormap[model.labels_],s=40)
plt.title( 'KMEANS Classfication')
plt.show()

print('Accuracy')
print(sm.accuracy_score(y,model.labels_))
print('Confusion_matrix')
print(sm.confusion_matrix(y,model.labels_))
print('classification_report')
print(sm.classification_report(y,model.labels_))

week 10

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
print(X)
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
print(y)

#Split the data into train and test samples
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.1)
print("Dataset is split into training and testing...")
print("Size of training data and its label",x_train.shape,y_train.shape)
print("Size of testing data and its label",x_test.shape,y_test.shape)

# prints Label no. and their names
for i in range(len(iris.target_names)):
  print("Label", i , "-",str(iris.target_names[i]))

#create object of KNN classifer
classifer = KNeighborsClassifier(n_neighbors=3)

#perform Training
classifer.fit(x_train, y_train)#perform teating
y_pred=classifer.predict(x_test)

#Display the results
print("Results of Classification using K-nn with K=3")
for r in range(0,len(x_test)):
  print(" sample:", str(x_test[r]), " Actual-label:",str(y_test[r]), " predict-label:", str(y_pred[r]))
print("Classification Accuracy :" , classifer.score(x_test,y_test))
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Ketrics')
print(classification_report(y_test,y_pred))

week11

import numpy as np
X = np.array(([2,9],[1,5],[3,6])) #Hours Studied,Hours Slept
y=np.array(([92],[86],[89])) #Test Score
y=y/100 #Max Test Score is 100
#Sigmoid Function
def sigmoid(x):
return 1/(1+ np.exp(-x))
#Derivatives of Sigmoid function
def derivatives_sigmoid(x):
return x*(1-x)
#Variable initialization
epoch=10000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayers_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons of output layer
#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayers_neurons))
bias_hidden=np.random.uniform(size=(1,hiddenlayers_neurons)) #bias matri
weight_hidden=np.random.uniform(size=(hiddenlayers_neurons,output_neurons)) #weight matrix to the output layer
bias_output=np.random.uniform(size=(1,output_neurons)) #matrix to output layer
for i in range(epoch):
hinp1=np.dot(X,wh)
hinp=hinp1+ bias_hidden
hlayer_activation = sigmoid(hinp)
outinp1=np.dot(hlayer_activation,weight_hidden)
outinp = outinp1+bias_output
output = sigmoid(outinp)
EO = y-output
outgrad=derivatives_sigmoid(output)
d_output = EO * outgrad
EH = d_output.dot(weight_hidden.T)
hiddengrad=derivatives_sigmoid(hlayer_activation)
d_hiddenlayer = EH * hiddengrad
weight_hidden += hlayer_activation.T.dot(d_output) * lr
bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) * lr
wh += X.T.dot(d_hiddenlayer) * lr
bias_output += np.sum(d_output,axis=0,keepdims=True) *lr
print("Input: \n"+str(X))
print("Actual Output: \n"+str(y))
print("Predicted Output: \n",output)

week12

12) Write a program to implement Support Vector Machine.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data # Features: sepal length, sepal width, petal length, petal width
y = iris.target # Labels: three species of Iris
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the SVM classifier
svm_model = SVC(kernel='linear') # You can also try 'rbf', 'poly', etc.
# Train the SVM model on the training data
svm_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = svm_model.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
