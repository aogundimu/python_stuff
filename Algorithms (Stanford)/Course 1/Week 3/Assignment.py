#!/Applications/anaconda/bin/python
import sys
import numpy as np

total_comp = 0

def partition(A, l, r, p):

    global total_comp
    total_comp = total_comp + (r-l)

    pivot = A[p]
    ## Swap the pivot value with the first element in the
    ## array segment
    tmp = A[l]
    A[l] = A[p]
    A[p] = tmp

    i = l + 1

    ## i has to be to the left of the first element that is
    ## bigger than the pivot
    for j in range(l+1, r+1):
        #print(i, j)
        if A[j] < pivot:
            tmp = A[i]
            A[i] = A[j]
            A[j] = tmp
            i = i + 1

    tmp = A[i-1]
    A[i-1] = A[l]
    A[l] = tmp
    # The position of the pivot after partioning
    return  i-1


def createArrayFromFile(input_file):
    result = []
    count = 0
    with open(input_file) as f:
        for line in f:
            count = count + 1
            result.append( int(line) )

    return result

#
# Quicksort using the first element as pivot
#
def quickSortF(A, p, r):
    #print("quicksort")
    #total_comp = total_comp + (r - p)
    #global total_comp
    #total_comp = total_comp + (r-p)

    if p < r:
        q = partition(A,p,r,p)
        quickSortF(A, p, q-1 )
        quickSortF(A, q+1, r)

###############################

#
# Quicksort using the last element as pivot
#
def quickSortL(A, p, r):
    #global total_comp
    #total_comp = total_comp + (r-p)

    if p < r:
        q = partition(A,p,r,r)
        quickSortF(A, p, q-1 )
        quickSortF(A, q+1, r)

#
# Quicksort using the median-of-3 elements as pivot
#
def quickSortM(A, p, r):
    #global total_comp
    #total_comp = total_comp + (r-p)

    if p < r:
        m = int((p+r)/2)
        med = int(np.median( [A[p], A[r], A[m] ]))
        s = 0
        if med == A[p]:
            s = p
        elif med == A[r]:
            s = r
        else:
            s = med

        q = partition(A,p,r,s)
        quickSortF(A, p, q-1 )
        quickSortF(A, q+1, r)
####
## 1. the first value = 162085 /// okay
## 2. Middle value = 158773
## 3. Last value = 159750
my_array = [3, 8, 2, 5, 1, 4, 7, 6, 9, 10, 11, 20, 19, 14, 77, 30, \
            18, 62, 99, 109, 203, 25]
#print(my_array)
#partition(my_array, 0, 8)
#print(my_array)
## Open the file and create and array from the content
if ( len(sys.argv) < 2 ):
    print( "Usage; " )
    print( "assignment.py fileName")
else :
    fileName = sys.argv[1]
    #print("filename = ", fileName )
    input_array = createArrayFromFile(fileName)
    #quickSortF(input_array, 0, len(input_array)-1)
    #quickSortL(input_array, 0, len(input_array)-1)
    quickSortM(input_array, 0, len(input_array)-1)
    #quickSortM(my_array, 0, len(my_array)-1)
    #print(input_array)
    print(total_comp)
