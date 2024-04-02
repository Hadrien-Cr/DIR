import numpy as np

def is_pure(y):
    # check if a collection of sample is pure
    return len(np.unique(y)) == 1

def gini_index(y):
    classes = np.unique(y)
    n_samples = len(y)
    gini = 0.0
    for c in classes:
        p = np.sum(y == c) / n_samples
        gini += p * (1 - p)
    return gini

def evaluate_split(x_feature_values, y, split_val):
    '''
    Evaluate the impurity reduction of a split of the 1d array x_feature_values, labelled by y, with the split value split_val
    Note that the gini index of the parent node is ommited, because it is a term that does not depend on the split.
    '''

    left_indices = x_feature_values <= split_val
    right_indices = x_feature_values > split_val
    
    left_gini = gini_index(y[left_indices])
    right_gini = gini_index(y[right_indices])
    
    n_samples = len(y)
    # Compute impurity reduction as the difference between the parent node's impurity and the weighted sum of child node impurities
    impurity_reduction = gini_index(y) - ((np.sum(left_indices) / n_samples) * left_gini + (np.sum(right_indices) / n_samples) * right_gini)
    
    return impurity_reduction

class CustomDecisionTreeClassifier:
    '''
    The tree is a binary tree classifier that uses all the data available
    '''
    def __init__(self,max_features=None,max_depth=None,min_samples_split=2,min_samples_leaf=1):
        '''
        the list features is all the features available to make split ie features=[0,2] means that the 1st feature and the 3rd feature of x are available
        We encode the tree as a list of size 2^(max_depth+1)-1
        where the children nodes of the node i are the nodes 2*i+1 and 2*i+2
        '''
        assert max_features>0
        assert min_samples_leaf >0
        assert min_samples_split>1
        
        self.max_features=max_features
        self.max_depth=max_depth
        self.max_nodes=int(2**(max_depth+1)-1)
        self.is_leaf_node=[False for i in range(self.max_nodes)] #store what are the leaf nodes
        self.is_split_node=[False for i in range(self.max_nodes)] #store what are the split nodes
        self.output_nodes=[None for i in range(self.max_nodes)] #output value on the specific leaf
        self.splits=[None for i in range(self.max_nodes)] #Each split node is encoded by a tuple (k,val): the split is made on feature k and the split value is val
        self.all_features=None
        self.min_samples_leaf=min_samples_leaf
        self.min_samples_split=min(min_samples_split,2*min_samples_leaf)


    def fit(self,x,y):
        self.all_features=[i for i in range(len(x[0]))]
        assert len(x)==len(y)
        self.update(0,x,y)
    
    def update(self,i,x,y):

        '''
        To fit the tree, each node i is updated one by one with the recursive rule: 

        Given the data (x,y) on the node i:
        case A: if possible, make this node a "split node"
                    -> find the best split for the data
                    -> call the update on the 2 children nodes and on the splitted data
        case B: else if there are not enough samples or if the max depth is reach or the node is pure, the node is a leaf node
                    -> change the output of this node to the most common value of y
        '''
        

        if len(y)>=self.min_samples_split and 2*i+1<self.max_nodes and not is_pure(y):
            self.is_split_node[i]=True
            #compute the best split
            
            # Randomly select features
            selected_features = np.random.choice(self.all_features,self.max_features,replace=False)

            #Assess each feature's ability to split
            best_feature,overall_best_split_val,overall_best_split_score= 0,0,-1 # this is the best split for all features k (initialized)
            
            for k in selected_features:
                x_feature_values=x[:,k]
                sorted_x_feature_values=sorted(list(set(x_feature_values)))
                split_vals=list(set([0.5*(sorted_x_feature_values[j+1]+sorted_x_feature_values[j]) for j in range(len(sorted_x_feature_values)-1)])) # all the possible splits

                best_split_val,best_split_score= None, -1 # this is the best split for the feature k (initialized)

                for split_val in split_vals:
                    score=evaluate_split(x_feature_values,y,split_val) 
                    if best_split_val==None or score>best_split_score:
                        best_split_val,best_split_score= split_val, score 

                if best_split_score>overall_best_split_score:
                    best_feature,overall_best_split_val,overall_best_split_score=k,best_split_val,best_split_score

            #store the split
            self.splits[i]=(best_feature,overall_best_split_val)
                        
            #split the data
            samples_left=x[:, best_feature] > overall_best_split_val
            samples_right=x[:, best_feature] <= overall_best_split_val

            x_left,y_left=x[samples_left],y[samples_left]
            x_right,y_right=x[samples_right],y[samples_right]
            
            #make the recursive update on the children nodes
            assert len(y_left)>0 
            assert len(y_right)>0 
            self.update(2*i+2,x_left,y_left)
            self.update(2*i +1,x_right,y_right)

        else:
            self.is_leaf_node[i]=True
            most_common_label=np.argmax(np.bincount(y))
            self.output_nodes[i]=most_common_label

    def predict(self,x):

        '''
        x is the array of the n samples we want to predict
        For each sample, start from the root node, and follow the splits until a leaf is reached
        '''
        n=len(x)
        predictions=np.zeros(n)
        for id in range(n):
            i=0
            while not self.is_leaf_node[i]:
                k,split_val=self.splits[i]
                if x[id,k]<= split_val:
                    i=2*i+1
                else:
                    i=2*i+2

            predictions[id]=self.output_nodes[i]
        return(predictions)





class CustomRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):

        assert max_features>0
        assert min_samples_leaf >0
        assert min_samples_split>1

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.estimators = []
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features)) # Default is square root of total features
        
        for _ in range(self.n_estimators):
            # Randomly select samples with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_X = X[indices]
            bootstrap_y = y[indices]
            

            
            # Create and fit a decision tree
            tree = CustomDecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf)
            tree.fit(bootstrap_X, bootstrap_y)
            self.estimators.append(tree)
    
    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        for i, tree in enumerate(self.estimators):
            predictions[:, i] = tree.predict(X)
        
        # Use majority voting to make final predictions
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions.astype(int))
    
    
'''

TEST SECTION

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def test_model(model,model_name,n_samples=1000, centers=3, n_features=2,make_plot=True,compute_accuracy=True):

    '''
    The dataset is trained on half of the samples
    Use make_plot=True to plot the decision boundary of the models
    Use compute_accuracy=True to output the accuracy of the test set
    '''
        
    # Generate data
    x, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=0)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
    # Train decision tree classifier
    model.fit(x_train, y_train)

    if make_plot:
        # Define a meshgrid within the range of data points
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Predict labels for each point in the meshgrid

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')

        # Plot data points
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary of: '+model_name)
        plt.show()

    if compute_accuracy:
        #Evaluate Accuracy
        acc=accuracy_score(model.predict(x_test),y_test)
        return(acc)
    

# Perform multiple experiments for each model
num_experiments = 10

results = {'tree': [],
           'rfc_custom': [],
           'rfc_sk': []}

for _ in range(num_experiments):
    # Perform experiments for each model
    tree = CustomRandomForestClassifier(max_features=1, n_estimators=1, max_depth=2)
    results['tree'].append(test_model(tree, 'tree', n_samples=1000, centers=3, n_features=2, make_plot=False, compute_accuracy=True))

    rfc_custom = CustomRandomForestClassifier(max_features=1, n_estimators=100, max_depth=2)
    results['rfc_custom'].append(test_model(rfc_custom, 'rfc_custom', n_samples=1000, centers=3, n_features=2, make_plot=False, compute_accuracy=True))

    rfc_sk = RandomForestClassifier(max_features=1, n_estimators=100, max_depth=2)
    results['rfc_sk'].append(test_model(rfc_sk, 'rfc_sk', n_samples=1000, centers=3, n_features=2, make_plot=False, compute_accuracy=True))

# Calculate average accuracy for each model
avg_results = {model: np.mean(acc_list) for model, acc_list in results.items()}

# Create bar chart
models = list(avg_results.keys())
accuracies = list(avg_results.values())

plt.barh(models, accuracies, color=['skyblue', 'salmon', 'lightgreen'])
plt.xlabel('Accuracy')
plt.title('Comparison of Model Performance')
# Add annotations
for i, acc in enumerate(accuracies):
    plt.text(acc, i, f'{acc:.2f}', ha='left', va='center')
plt.show()
