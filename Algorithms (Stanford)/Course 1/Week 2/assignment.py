#!/Applications/anaconda/bin/python

import sys
import array




def countInversionBruteForce(input_array):
    count = 0
    length = len(input_array);
    for i in range(length):
        for j in range(i+1, length):
            if input_array[i] > input_array[j]:
                count = count + 1

    return count


def sortAndCount(array1, array2):
    return 0

def countInverRecursive(input_array):

    if (len(input_array) < 2):
        return 0
    # split array into two
    
    return 0

def createArrayFromFile(input_file):
    result = []
    count = 0
    with open(input_file) as f:
        for line in f:
            #count = count + 1
            result.append( int(line))

    return result

################
#print("This is the name of the script: ", sys.argv[0])
#print("Number of arguments: ", len(sys.argv) )
#print("The arguments are: " , str(sys.argv) )
if ( len(sys.argv) < 2 ):
    print( "Usage; " )
    print( "assignment.py fileName")
else :
    fileName = sys.argv[1]
    #print("filename = ", fileName )
    input_array = createArrayFromFile(fileName)
    print( countInversionBruteForce(input_array) )
    print( countInverRecursive(input_array) )
