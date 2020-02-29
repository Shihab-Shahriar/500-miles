# one row c1 represents format of the parse_classification_report(report_gen) 
# 4 numbers for each class [ precision, recall, f1score, support]
#  In our case, clusters will have a list of 10 of those reports summarized
#==========================================================================
c1 = {"1" : [1, 2, 3, 4], "2": [2,3,4, 5], "3": [3,4,5,6], "4": [4,5, 6, 7], "avg": [5,6,7, 8]}
c2 = {"1" : [0, 1, 2, 3], "2": [0,1,2, 3], "3": [0, 1, 2, 3], "4": [0, 1, 2, 3], "avg": [0, 1, 2, 3]}
c3 = {"1" : [1, 2, 3, 4], "2": [2,3,4, 5], "3": [3,4,5,6], "4": [4,5, 6, 7], "avg": [5,6,7, 8]}
c4 = {"1" : [0, 1, 2, 3], "2": [0,1,2, 3], "3": [0, 1, 2, 3], "4": [0, 1, 2, 3], "avg": [0, 1, 2, 3]}
c5 = {"1" : [1, 2, 3, 4], "2": [2,3,4, 5], "3": [3,4,5,6], "4": [4,5, 6, 7], "avg": [5,6,7, 8]}
clusters = []
clusters.append(c1)
clusters.append(c2)
clusters.append(c3)
clusters.append(c4)
clusters.append(c5)
#========================================================================================================

# summarize here
total_summary = {'1': [], '2': [], '3': [], '4': [], 'avg': []}
key_values = ['1', '2', '3', '4', 'avg']

for c in clusters:
    for i in key_values:
        list_temp = []  # get the values from the matrix
        for j in range(4): # 4 numbers [ precision, recall, f1score, support]
            list_temp.append(c[i][j])
        if(len(total_summary[i]) == 0):
            total_summary[i] = list_temp # empty list initialized
        else:
            for k in range(4): # sum corresponding elements
                total_summary[i][k] = total_summary[i][k]  + list_temp[k]

print(total_summary)
        

    