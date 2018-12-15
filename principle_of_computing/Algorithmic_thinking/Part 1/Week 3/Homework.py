#!/Applications/anaconda/bin/python


def question1():

    graph = [[1,0,0],[0,0,0],[0,0,0]]
    X = set([])
    
    for i in range(0, len(graph)):
        flag = True
        for j in range(0, len(graph[i])):
            #print( graph[0][j] )
            if graph[i][j] == 1:
                flag = False
                break

        if flag == True:
            print('flag is tru')
            X = X.union(set([i]))
            print(X)

    return X
                       
##############

print( question1() )
