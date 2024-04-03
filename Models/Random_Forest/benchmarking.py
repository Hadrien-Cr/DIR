from Random_Forest_Classifier import *
import numpy as np
from sklearn.model_selection import train_test_split

'''
Retrieve the data 
Source: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
'''

x,y=np.zeros((699,9)),np.zeros((699),dtype='int64')
with open('Models//Random_Forest//breast+cancer+wisconsin+original//breast-cancer-wisconsin.data', 'r') as file:
    contents = file.read()
    lines = contents.splitlines()
    for row,line in enumerate(lines):
        if row<700:
            l=line.split(',')
            for column in range(0,9):
                x[row,column]=(l[column+1] if l[column+1]!= '?' else -1)
            y[row]=(1 if l[10]=='4' else 0)


'''
Make the experiment
'''
if __name__ =='__main__':
    # Perform multiple experiments for each model
    num_experiments = 50

    results = {'tree': [],
            'rfc_custom': [],
            'rfc_sk': []}

    
    for _ in tqdm(range(num_experiments)):

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
        
        # Performance of the tree
        tree = CustomDecisionTreeClassifier(max_features=1, max_depth=11)
        tree.fit(x_train,y_train)
        tree.predict(x)
        acc_tree=accuracy_score(tree.predict(x_test),y_test)
        results['tree'].append(acc_tree)

        # Performance of the custom rfc
        rfc_custom=CustomRandomForestClassifier(max_features=1,n_estimators=100,max_depth=11)
        rfc_custom.fit(x_train,y_train)
        rfc_custom.predict(x)
        acc_rfc_custom=accuracy_score(rfc_custom.predict(x_test),y_test)
        results['rfc_custom'].append(acc_rfc_custom)

        # Performance of the custom rfc
        rfc_sk=RandomForestClassifier(max_features=1,n_estimators=100,max_depth=11)
        rfc_sk.fit(x_train,y_train)

        acc_rfc_sk=accuracy_score(rfc_sk.predict(x_test),y_test)
        
        results['rfc_sk'].append(acc_rfc_sk)


    # Calculate average accuracy for each model
    avg_results = {model: np.mean(acc_list) for model, acc_list in results.items()}
    benchmark_accuracy = 0.973

    # Create bar chart
    models = list(avg_results.keys())
    accuracies = list(avg_results.values())
    plt.figure(figsize=(6, 4))  # Adjusted figure size for better visibility
    plt.barh(models, accuracies, color=['skyblue', 'salmon', 'lightgreen'])
    plt.axvline(x=benchmark_accuracy, color='gray', linestyle='--', label='Benchmark (0.973)')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Model Performance')
    plt.legend()  # Added legend for the benchmark line
    # Add annotations
    for i, acc in enumerate(accuracies):
        plt.text(acc-0.15, i, f'{acc:.4f}', ha='center', va='center')
    plt.savefig('Models//Random_Forest//Benchmarking_rfc.png')

    plt.show()