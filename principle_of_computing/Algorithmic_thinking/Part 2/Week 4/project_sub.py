"""
This module implements the algorithms for doing the following:
1. Create a Scoring Matrix for sequence alignment
2. Create an alignment matrix for sequence alignment
3. Implementation of the global alignment algorithm
4. Implementation of the local alignment algorithm
"""
#############################################################################
###
def build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score):
    """
    The function creates a scoring matrix in the form a dict of dicts. This is
    based on the values specified in the arguments.
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

#############################################################################
###
def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag):
    """
    This method creates a 2-dimensional matrix denoting the alignment matrix
    for the two sequences provided. 
    """

    len_seq_x = len(seq_x)
    len_seq_y = len(seq_y)

    result_matrix = [[0 for _ in range(len_seq_y+1)] for _ in range(len_seq_x+1)]

    cur_val = 0
    for index in range(1, len_seq_x+1):
        value = scoring_matrix.get(seq_x[index-1]).get('-')
        cur_val += value       
        result_matrix[index][0] = cur_val
        if global_flag:
            result_matrix[0][index] = cur_val
        else:
            result_matrix[0][index] = max(0, cur_val)
        
    cur_val = 0
    for index in range(1, len_seq_y+1):
        value = scoring_matrix.get('-').get(seq_y[index-1])
        cur_val += value
        if global_flag:
            result_matrix[0][index] = cur_val
        else:
            result_matrix[0][index] = max(0, cur_val)
            
    for ind_x in range(1, len_seq_x+1):
        for ind_y in range(1, len_seq_y+1):
            value1 = result_matrix[ind_x-1][ind_y-1] + \
                     scoring_matrix.get(seq_x[ind_x-1]).get(seq_y[ind_y-1])
                
            value2 = result_matrix[ind_x-1][ind_y]  + \
                     scoring_matrix.get(seq_x[ind_x-1]).get('-')
                
            value3 = result_matrix[ind_x][ind_y-1] +  \
                     scoring_matrix.get('-').get(seq_y[ind_y-1])
            if global_flag:
                result_matrix[ind_x][ind_y] = max(value1, value2, value3)
            else:
                result_matrix[ind_x][ind_y] = max(0, value1, value2, value3)
            
    return result_matrix


#############################################################################
###
def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    This method is an implementation of the global alignment algorithm for 
    sequences.
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
        aligned_x = seq_x[len_x-1] + aligned_x
        aligned_y = "-" + aligned_y
        len_x -= 1

    while len_y != 0:
        aligned_x = "-" + aligned_x
        aligned_y = seq_y[len_y-1] + aligned_y
        len_y -= 1

    return (alignment_matrix[len(seq_x)][len(seq_y)], aligned_x, aligned_y)


#############################################################################
###
def find_maximum_value_index(matrix):
    """
    This method finds the maximum value and its index in a 2-d array
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
    This method implements the local alignment algorithm.
    """

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
    


