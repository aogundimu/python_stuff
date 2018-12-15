#!/Applications/anaconda/bin/python

"""
This is the grid stuff.

Given a square, we can store the index associated with a square as the 
tuple (row,col) in Python. Then, we can represent some relevant property 
of that square as the entry 𝚌𝚎𝚕𝚕𝚜[𝚛𝚘𝚠][𝚌𝚘𝚕] in the 2D list 𝚌𝚎𝚕𝚕𝚜. Note the 
expression 𝚌𝚎𝚕𝚕𝚜[𝚛𝚘𝚠] returns a 1D list corresponding to a row of the grid. 
We can initialize this 2D list via the code fragment:

cells = [ [... for col in range(grid_width)] for row in range(grid_height)]
"""

import numpy as np

#############################################################################
###
def build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score):
    """
    For scoring matrices, we take a different approach since the rows and 
    the columns of the matrix are indexed by characters in Σ∪{′−′}. In particular, 
    we will represent a scoring matrix in Python as a dictionary of dictionaries. 
    Given two characters 𝚛𝚘𝚠_𝚌𝚑𝚊𝚛 and 𝚌𝚘𝚕_𝚌𝚑𝚊𝚛, we can access the matrix entry 
    corresponding to this pair of characters via 𝚜𝚌𝚘𝚛𝚒𝚗𝚐_𝚖𝚊𝚝𝚛𝚒𝚡[𝚛𝚘𝚠_𝚌𝚑𝚊𝚛][𝚌𝚘𝚕_𝚌𝚑𝚊𝚛].
    """

    local_alphabet = alphabet.union( set(['-']) )
    
    result = {}

    for char in local_alphabet:
        curr_dict = {}
        for sec_char in local_alphabet:
            if sec_char == char:
                if sec_char == '-':
                    curr_dict.update({sec_char: dash_score})
                else:
                    curr_dict.update({sec_char: diag_score})
            elif sec_char == '-':
                curr_dict.update({sec_char: dash_score})
            else:
                if char == '-':
                    curr_dict.update({sec_char: dash_score})
                else:
                    curr_dict.update({sec_char: off_diag_score})

        result.update({char: curr_dict})
        
    return result

def test_build_scoring_matrix():

    alphabet = set(['A', 'T', 'C', 'G', '-'])
    diag_score = 10
    off_diag_score = 5
    dash_score = -5    

    #print( build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score) )

    print( build_scoring_matrix(set(['A', 'C', 'T', 'G']), 6, 2, -4) )
    
#############################################################################
###
def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag):
    """
    Alignment matrices will follow the same indexing scheme that we used for 
    grids in "Principles of Computing". Entries in the alignment matrix will 
    be indexed by their row and column with these integer indices starting at 
    zero. We will model these matrices as lists of lists in Python and can access 
    a particular entry via an expression of the form 𝚊𝚕𝚒𝚐𝚗𝚖𝚎𝚗𝚝_𝚖𝚊𝚝𝚛𝚒𝚡[𝚛𝚘𝚠][𝚌𝚘𝚕].

    𝚌𝚘𝚖𝚙𝚞𝚝𝚎_𝚊𝚕𝚒𝚐𝚗𝚖𝚎𝚗𝚝_𝚖𝚊𝚝𝚛𝚒𝚡(𝚜𝚎𝚚_𝚡, 𝚜𝚎𝚚_𝚢, 𝚜𝚌𝚘𝚛𝚒𝚗𝚐_𝚖𝚊𝚝𝚛𝚒𝚡, 𝚐𝚕𝚘𝚋𝚊𝚕_𝚏𝚕𝚊𝚐): 
    Takes as input two sequences 𝚜𝚎𝚚_𝚡 and 𝚜𝚎𝚚_𝚢 whose elements share a common 
    alphabet with the scoring matrix 𝚜𝚌𝚘𝚛𝚒𝚗𝚐_𝚖𝚊𝚝𝚛𝚒𝚡. The function computes and 
    returns the alignment matrix for 𝚜𝚎𝚚_𝚡 and 𝚜𝚎𝚚_𝚢 as described in the Homework. 
    If 𝚐𝚕𝚘𝚋𝚊𝚕_𝚏𝚕𝚊𝚐 is 𝚃𝚛𝚞𝚎, each entry of the alignment matrix is computed using 
    the method described in Question 8 of the Homework. If 𝚐𝚕𝚘𝚋𝚊𝚕_𝚏𝚕𝚊𝚐 is 𝙵𝚊𝚕𝚜𝚎, 
    each entry is computed using the method described in Question 12 of the Homework.
    """

    len_seq_x = len(seq_x)
    len_seq_y = len(seq_y)

    # This initializes result_matrix[0][0] to 0.
    # As a matter of fact it did it for all of the entries
    # result_matrix = [[0 for x in range(len_seq_y+1)] for y in range(len_seq_x+1)]
    result_matrix = [[0]*5 for i in range(5)]
    result_matrix = [[0 for _ in range(len_seq_y+1)] for _ in range(len_seq_x+1)]

    if global_flag:
        cur_val = 0
        for index in range(1, len_seq_x+1):
            value = scoring_matrix.get(seq_x[index-1]).get('-')
            cur_val += value
            result_matrix[index][0] = cur_val
 
        cur_val = 0
        for index in range(1, len_seq_y+1):
            value = scoring_matrix.get('-').get(seq_y[index-1])
            cur_val += value
            result_matrix[0][index] = cur_val 
 
        for ind_x in range(1, len_seq_x+1):
            for ind_y in range(1, len_seq_y+1):
                value1 = result_matrix[ind_x-1][ind_y-1] + \
                         scoring_matrix.get(seq_x[ind_x-1]).get(seq_y[ind_y-1])
                
                value2 = result_matrix[ind_x-1][ind_y]  + \
                         scoring_matrix.get(seq_x[ind_x-1]).get('-')
                
                value3 = result_matrix[ind_x][ind_y-1] +  \
                         scoring_matrix.get('-').get(seq_y[ind_y-1])
                result_matrix[ind_x][ind_y] = max(value1, value2, value3)

    else:
        cur_val = 0
        for index in range(1, len_seq_x+1):
            value = scoring_matrix.get(seq_x[index-1]).get('-')
            cur_val += value
            if cur_val < 0:
                result_matrix[index][0] = 0
            else:
                result_matrix[index][0] = cur_val            

        cur_val = 0
        for index in range(1, len_seq_y+1):
            value = scoring_matrix.get('-').get(seq_y[index-1])
            cur_val += value
            if cur_val < 0:
                result_matrix[0][index] = 0
            else:
                result_matrix[0][index] = cur_val + value

        for ind_x in range(1, len_seq_x+1):
            for ind_y in range(1, len_seq_y+1):
                value1 = result_matrix[ind_x-1][ind_y-1] + \
                         scoring_matrix.get(seq_x[ind_x-1]).get(seq_y[ind_y-1])
                
                value2 = result_matrix[ind_x-1][ind_y]  + \
                         scoring_matrix.get(seq_x[ind_x-1]).get('-')
                
                value3 = result_matrix[ind_x][ind_y-1] +  \
                         scoring_matrix.get('-').get(seq_y[ind_y-1])
                max_val = max(value1, value2, value3)
                if max_val < 0:
                    result_matrix[ind_x][ind_y] = 0
                else:
                    result_matrix[ind_x][ind_y] = max_val
            
    return result_matrix


        
        
def test_compute_alignment_matrix():
    """
    """
    alphabet = set(['A', 'T', 'C', 'G', '-'])
    diag_score = 10
    off_diag_score = 5
    dash_score = -5
    score_matrix = build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score)
    # seq_x = ['A','C']
    # seq_y = ['T','A','G']
    seq_x = "AC"
    seq_y = "TAG"
    global_flag = True
    align_mat = compute_alignment_matrix(seq_x, seq_y, score_matrix, global_flag)
    print(align_mat)

#############################################################################
###
def compute_score(aligned_x, aligned_y, scoring_matrix):
    """
    This function calculates the score for the alignment of the two sequences using 
    the provided scoring_matrix.

    It takes as input 2 aligned sequences and the scoring matrix.
    It returns the score of the alignment or zero if the the two sequences are not the 
    same length
    """
    total_score = 0

    if ( len(aligned_x) == len(aligned_y) ):
        for index in range(0, len(aligned_x)):
            total_score += scoring_matrix.get(aligned_x[index]).get(aligned_y[index])
        return total_score
    else:
        return 0    
   
    
#############################################################################
###
def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    The first function will implement the method ComputeAlignment discussed in
    Question 9 of the Homework.

    This function computes a global alignment of 𝚜𝚎𝚚_𝚡 and 𝚜𝚎𝚚_𝚢 using the global 
    alignment matrix 𝚊𝚕𝚒𝚐𝚗𝚖𝚎𝚗𝚝_𝚖𝚊𝚝𝚛𝚒𝚡.The function returns a tuple of the form 
    (𝚜𝚌𝚘𝚛𝚎, 𝚊𝚕𝚒𝚐𝚗_𝚡, 𝚊𝚕𝚒𝚐𝚗_𝚢) where 𝚜𝚌𝚘𝚛𝚎 is the score of the global alignment 
    𝚊𝚕𝚒𝚐𝚗_𝚡 and 𝚊𝚕𝚒𝚐𝚗_𝚢.     
    """

    len_x = len(seq_x)
    len_y = len(seq_y)

    aligned_x = ""
    aligned_y = ""

    while len_x != 0 and len_y != 0:
        value = alignment_matrix[len_x-1][len_y-1] + \
                scoring_matrix.get(seq_x[len_x-1]).get(seq_y[len_y-1])
        if alignment_matrix[len_x][len_y] == value:
            aligned_x = seq_x[len_x-1] + aligned_x
            aligned_y = seq_y[len_y-1] + aligned_y
            len_x -= 1
            len_y -= 1
        else:
            value = alignment_matrix[len_x-1][len_y] + \
                    scoring_matrix.get(seq_x[len_x-1]).get('-')
            if alignment_matrix[len_x][len_y] == value:
                aligned_x = seq_x[len_x-1] + aligned_x
                aligned_y = "-" + aligned_y
                len_x -= 1
            else:
                aligned_x = "-" + aligned_x
                aligned_y = seq_y[len_y-1] + aligned_y
                len_y -= 1

    while len_x != 0:
        aligned_x = sec_x[len_x-1] + aligned_x
        aligned_y = "-" + aligned_y
        len_x -= 1

    while len_y != 0:
        aligned_x = "-" + aligned_x
        aligned_y = sec_y[len_y-1] + aligned_y
        len_y -= 1

    return (alignment_matrix[len(seq_x)][len(seq_y)], aligned_x, aligned_y)


#############################################################################
###
def find_maximum_value_index(matrix):
    """
    """

    max_value = matrix[0][0]
    row_index = 0
    column_index = 0

    for row in range(0, len(matrix)):
        for column in range(0, len(matrix[row])):
            if matrix[row][column] > max_value:
                max_value = matrix[row][column]
                row_index = row
                column_index = column

    return (row_index, column_index)

#############################################################################
###
def compute_local_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    Takes as input two sequences 𝚜𝚎𝚚_𝚡 and 𝚜𝚎𝚚_𝚢 whose elements share a common 
    alphabet with the scoring matrix 𝚜𝚌𝚘𝚛𝚒𝚗𝚐_𝚖𝚊𝚝𝚛𝚒𝚡. This function computes a 
    local alignment of 𝚜𝚎𝚚_𝚡 and 𝚜𝚎𝚚_𝚢 using the local alignment matrix 
    𝚊𝚕𝚒𝚐𝚗𝚖𝚎𝚗𝚝_𝚖𝚊𝚝𝚛𝚒𝚡. The function returns a tuple of the form 
    (𝚜𝚌𝚘𝚛𝚎, 𝚊𝚕𝚒𝚐𝚗_𝚡, 𝚊𝚕𝚒𝚐𝚗_𝚢) where 𝚜𝚌𝚘𝚛𝚎 is the score of the optimal local 
    alignment 𝚊𝚕𝚒𝚐𝚗_𝚡 and 𝚊𝚕𝚒𝚐𝚗_𝚢.

    During the computation of a global alignment, the traceback in the global 
    alignment matrix S starts at the bottom right corner of the matrix (S[m,n]) 
    and traces to the upper left corner (S[0,0]). Given the local alignment 
    matrix S computed in Question 12, Algorithm ComputeAlignment (in Question 9) 
    can be modified to generate a local alignment of two sequences as follows:

    Start the traceback from the entry in S that has the maximum value over the 
    entire matrix and trace backwards using exactly the same technique as in 
    ComputeAlignment. Stop the traceback when the first entry with value 0 is 
    encountered. If the local alignment matrix has more than one entry that has 
    the maximum value, any entry with maximum value may be used as the starting entry.
    """

    # find the maximum value from the alignment_matrix
    max_value_index = find_maximum_value_index(alignment_matrix)

    aligned_x = ""
    aligned_y = ""

    len_x = max_value_index[0]
    len_y = max_value_index[1] 

    while (len_x  != 0 and len_y !=0) and alignment_matrix[len_x][len_y] != 0:
        value = alignment_matrix[len_x-1][len_y-1] + \
                scoring_matrix.get(seq_x[len_x-1]).get(seq_y[len_y-1])
        if alignment_matrix[len_x][len_y] == value:
            aligned_x = seq_x[len_x-1] + aligned_x
            aligned_y = seq_y[len_y-1] + aligned_y
            len_x -= 1
            len_y -= 1
        else:
            value = alignment_matrix[len_x-1][len_y] + \
                    scoring_matrix.get(seq_x[len_x-1]).get('-')
            if alignment_matrix[len_x][len_y] == value:
                aligned_x = seq_x[len_x-1] + aligned_x
                aligned_y = "-" + aligned_y
                len_x -= 1
            else:
                aligned_x = "-" + aligned_x
                aligned_y = seq_y[len_y-1] + aligned_y
                len_y -= 1

    return(alignment_matrix[max_value_index[0]][max_value_index[1]], \
           aligned_x, aligned_y)
    
#############################################################################
###
def test_grid():
    
    """
    This is the result of the following algorithm
    [[1, 2, 3, 4, 5], 
     [6, 7, 8, 9, 10], 
     [11, 12, 13, 14, 15], 
     [16, 17, 18, 19, 20], 
     [21, 22, 23, 24, 25] ]
    """
    cells = [ [... for col in range(4)] for row in range(3)]
    
    count = 1
    
    for i in range(0, len(cells)):
        row = cells[i]
        for j in range(0, len(row)):
            row[j] = count
            count += 1    

    print(cells)
    print(cells[2][0])

#############################################################################
###
#test_build_scoring_matrix()
#test_grid()
test_compute_alignment_matrix()
# print( max(1, 2, 3) )
#twod = [ [3,7,20], [7, 1, 30], [22,18,10] ]
#print( find_maximum_value_index(twod) )
