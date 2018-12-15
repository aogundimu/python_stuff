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
DESKTOP = True

if DESKTOP:
    import matplotlib.pyplot as plt
#    import alg_project4_solution as student
else:
    import simpleplot
    import userXX_XXXXXXX as student
    
import numpy as np
import math
import random
import urllib
import copy



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
# provided code
def read_scoring_matrix(filename):
    """
    Read a scoring matrix from the file named filename.  

    Argument:
    filename -- name of file containing a scoring matrix

    Returns:
    A dictionary of dictionaries mapping X and Y characters to scores
    """
    scoring_dict = {}
    #scoring_file = urllib2.urlopen(filename)
    scoring_file = urllib.request.urlopen(filename)
    ykeys = scoring_file.readline().decode("utf-8")
    ykeychars = ykeys.split()

    
    score_text = scoring_file.read().decode("utf-8")
    score_lines = score_text.split('\n')

    for line in score_lines:
        vals = line.split()
        xkey = vals.pop(0)
        scoring_dict[xkey] = {}
        for ykey, val in zip(ykeychars, vals):
            scoring_dict[xkey][ykey] = int(val)
    
    return scoring_dict

def read_protein(filename):
    """
    Read a protein sequence from the file named filename.

    Arguments:
    filename -- name of file containing a protein sequence

    Returns:
    A string representing the protein
    """
    #protein_file = urllib2.urlopen(filename)
    protein_file = urllib.request.urlopen(filename)
    protein_seq = protein_file.read().decode("utf-8")
    protein_seq = protein_seq.rstrip()
    return protein_seq


def read_words(filename):
    """
    Load word list from the file named filename.

    Returns a list of strings.
    """
    # load assets
    #word_file = urllib2.urlopen(filename)
    word_file = urllib.request.urlopen(filename)
    
    # read in files as string
    words = word_file.read().decode("utf-8")
    
    # template lines and solution lines list of line string
    word_list = words.split('\n')
    print ("Loaded a dictionary with", len(word_list), "words")
    return word_list

def remove_dashes(sequence):
    return sequence.replace("-", "")

def do_comparison( global_result ):
    string1 = global_result[1]
    string2 = global_result[2]

    string_length = len(string1)
    print("Sequence lengths = ",string_length)
    matches = 0

    for index in range(0, string_length):
        if string1[index] == string2[index]:
            matches += 1

    print("Total Matches = ", matches )
    return ( matches/string_length * 100) 

###############################################################
def generate_null_distribution(seq_x, seq_y, scoring_matrix, num_trials):
    """
    The horizontal axis should be the scores and the vertical axis should be the fraction of total 
    trials corresponding to each score.
    """
    local_seq_y = copy.deepcopy(seq_y)
    
    result = { }

    for x in range(0, num_trials):
        print("Loop number - ", x)
        alignment_matrix = compute_alignment_matrix(seq_x, local_seq_y, scoring_matrix, False)
        local_alignment = compute_local_alignment(seq_x, local_seq_y, scoring_matrix, alignment_matrix)
        score = local_alignment[0]
        total_so_far = result.get(score, -1)
        if total_so_far == -1:
            result.update({score: 1})
        else:
            result.update( {score: total_so_far+1} )
        lst = list(local_seq_y)
        random.shuffle(lst)
        local_seq_y = "".join(lst)

    return result

def normalize_distribution( distribution ):
    total_sum = 0
    for value in distribution.values():
        total_sum += value

    result = {}
    for key in distribution.keys():
        result.update({key: distribution.get(key)/total_sum})

    return result

#####################################################################
def plot_normalized_distribution( normalized_distr ):

    xaxis = []
    for value in normalized_distr.keys():
        xaxis.append(value)

    yaxis = []    
    for value in normalized_distr.values():
        yaxis.append(value)

    plt.bar(xaxis, yaxis)
    plt.xlabel('Scores')
    plt.ylabel('Fraction of Trials')
    plt.title('Null Distribution For 1000 Trials')
    plt.show()


def do_calculations( distribution ):

    total_count = 0
    total_score = 0
    for key in distribution.keys():
        count = distribution.get(key)
        total_count += count
        total_score += (count * key)

    
    mean = total_score / total_count

    #print("Total Score = ", total_score)
    print("Mean = ", mean)

    # std_dev
    total_count = 0
    total_sum = 0
    for key in distribution.keys():
        count = distribution.get(key)
        total_count += count
        for _ in range(0, count):
            total_sum += (key - mean)**2

    std_dev = math.sqrt(total_sum/total_count)
    print("Standard Deviation = ", std_dev)

    z_score = ( 875 - mean ) / std_dev
    print("Z score = ", z_score )

## (𝚌𝚑𝚎𝚌𝚔𝚎𝚍_𝚠𝚘𝚛𝚍, 𝚍𝚒𝚜𝚝, 𝚠𝚘𝚛𝚍_𝚕𝚒𝚜𝚝) 
def check_spelling( checked_word, dist, word_list ):
    
    
#############################################################################
###
#test_build_scoring_matrix()
#test_grid()
#test_compute_alignment_matrix()
# print( max(1, 2, 3) )
#twod = [ [3,7,20], [7, 1, 30], [22,18,10] ]
#print( find_maximum_value_index(twod) )

PAM50_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_PAM50.txt"
HUMAN_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_HumanEyelessProtein.txt"
FRUITFLY_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_FruitflyEyelessProtein.txt"
CONSENSUS_PAX_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_ConsensusPAXDomain.txt"
WORD_LIST_URL = "http://storage.googleapis.com/codeskulptor-assets/assets_scrabble_words3.txt"


########  Question 1.
#hep = read_protein( HUMAN_EYELESS_URL )
#print(hep)
#fep = read_protein( FRUITFLY_EYELESS_URL )
#print(fep)
#scoring_matrix = read_scoring_matrix( PAM50_URL )
#print(scoring_matrix)

#alignment_matrix = compute_alignment_matrix( hep, fep, scoring_matrix, False)
#local_alignment = compute_local_alignment(hep, fep, scoring_matrix, alignment_matrix )

########## Answer Question 1.
# (875,
# 'HSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATPEVVSKIAQYKRECPSIFAWEIRDRLLSEGVCTNDNIPSVSSINRVLRNLASEK-QQ',
# 'HSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATAEVVSKISQYKRECPSIFAWEIRDRLLQENVCTNDNIPSVSSINRVLRNLAAQKEQQ')

#####   Question 2.
#cpu = read_protein(CONSENSUS_PAX_URL)
seq_one = 'HSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATPEVVSKIAQYKRECPSIFAWEIRDRLLSEGVCTNDNIPSVSSINRVLRNLASEK-QQ'
seq_two = 'HSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATAEVVSKISQYKRECPSIFAWEIRDRLLQENVCTNDNIPSVSSINRVLRNLAAQKEQQ'

#seq_one_no_dashes = remove_dashes( seq_one )
#seq_two_no_dashes = remove_dashes( seq_two )

#alignment_matrix_1 = compute_alignment_matrix( seq_one_no_dashes, cpu, scoring_matrix, True )
#alignment_matrix_2 = compute_alignment_matrix( seq_two_no_dashes, cpu, scoring_matrix, True )

#global_alignment_1 = compute_global_alignment(seq_one_no_dashes, cpu, scoring_matrix, alignment_matrix_1)
#global_alignment_2 = compute_global_alignment(seq_two_no_dashes, cpu, scoring_matrix, alignment_matrix_2)

#percentage1 = do_comparison( global_alignment_1 )
#percentage2 = do_comparison( global_alignment_2 )

#print( percentage1, percentage2 )
#print( global_alignment_1 )
#print( global_alignment_2 )

########  Answer Question 2
# percentage1 = 72.93233082706767 percentage2 = 70.1492537313433
# (613, '-HSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATPEVVSKIAQYKRECPSIFAWEIRDRLLSEGVCTNDNIPSVSSINRVLRNLASEKQQ',
#       'GHGGVNQLGGVFVNGRPLPDVVRQRIVELAHQGVRPCDISRQLRVSHGCVSKILGRYYETGSIKPGVIGGSKPKVATPKVVEKIAEYKRQNPTMFAWEIRDRLLAERVCDNDTVPSVSSINRIIR--------')
# (586, '-HSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATAEVVSKISQYKRECPSIFAWEIRDRLLQENVCTNDNIPSVSSINRVLRNLAAQKEQQ',
#       'GHGGVNQLGGVFVNGRPLPDVVRQRIVELAHQGVRPCDISRQLRVSHGCVSKILGRYYETGSIKPGVIGGSKPKVATPKVVEKIAEYKRQNPTMFAWEIRDRLLAERVCDNDTVPSVSSINRIIR---------')


#####$ Question 3
## Answer Question 3
#  The matches are 97 out of 133 for the human and 94 out of 134 for the fruitfly. I am not very familiar with this field of study and as such
#  it is very hard for me to say whether the 73 and 70 percentages are very high or very low. I will however go out on a limb and say that these
#  percentages are very high and it is very unlikely that this is a coincidence.

######### Question 4
#result = generate_null_distribution(hep, fep, scoring_matrix, 1000)
#print(result)


### {64: 0.009, 65: 0.013, 66: 0.002, 67: 0.006, 68: 0.002, 69: 0.005, 70: 0.003, 71: 0.003, 72: 0.004, 73: 0.001, 77: 0.003, 78: 0.001, 43: 0.023, 82: 0.001, 84: 0.001, 85: 0.001, 91: 0.001, 38: 0.002, 39: 0.002, 40: 0.008, 41: 0.009, 42: 0.022, 875: 0.001, 44: 0.054, 45: 0.045, 46: 0.057, 47: 0.066, 48: 0.072, 49: 0.064, 50: 0.058, 51: 0.064, 52: 0.07, 53: 0.056, 54: 0.047, 55: 0.035, 56: 0.038, 57: 0.031, 58: 0.021, 59: 0.019, 60: 0.032, 61: 0.014, 62: 0.018, 63: 0.016}

#norm_distr = normalize_distribution({38: 2, 39: 2, 40: 8, 41: 9, 42: 22, 43: 23, 44: 54, 45: 45, 46: 57, 47: 66, 48: 72, 49: 64, 50: 58, 51: 64, 52: 70, 53: 56, 54: 47, 55: 35, 56: 38, 57: 31, 58: 21, 59: 19, 60: 32, 61: 14, 62: 18, 63: 16, 64: 9, 65: 13, 66: 2, 67: 6, 68: 2, 69: 5, 70: 3, 71: 3, 72: 4, 73: 1, 77: 3, 78: 1, 82: 1, 84: 1, 85: 1, 91: 1})

#print( norm_distr )
#plot_normalized_distribution( norm_distr )

###############  Question 5
do_calculations({38: 2, 39: 2, 40: 8, 41: 9, 42: 22, 43: 23, 44: 54, 45: 45, 46: 57, 47: 66, 48: 72, 49: 64, 50: 58, 51: 64, 52: 70, 53: 56, 54: 47, 55: 35, 56: 38, 57: 31, 58: 21, 59: 19, 60: 32, 61: 14, 62: 18, 63: 16, 64: 9, 65: 13, 66: 2, 67: 6, 68: 2, 69: 5, 70: 3, 71: 3, 72: 4, 73: 1, 77: 3, 78: 1, 82: 1, 84: 1, 85: 1, 91: 1})

#############   Answer to question 5
# Mean =  51.76276276276276
# Standard Deviation =  6.8798333469391775
# Z score =  119.65947367075303

############### Question 6
# The distribution is bell shaped which indicates a normal distribution. The z-score of almost 120 indicate that
# the local alignment score of 875 for the human eyeless protein vs. the fruitfly eyeless protein was not by chance.# The likelihood of winning the lottery is much much higher than this.

###############  Question 7
# 𝚍𝚒𝚊𝚐_𝚜𝚌𝚘𝚛𝚎 is exactly 2, 𝚘𝚏𝚏_𝚍𝚒𝚊𝚐_𝚜𝚌𝚘𝚛𝚎 is exactly 1, 𝚍𝚊𝚜𝚑_𝚜𝚌𝚘𝚛e is exactly 0.

##############   Question 8
# ['bumble', 'fumble', 'humble', 'humbled', 'humbler', 'humbles', 'humbly','jumble', 'mumble', 'rumble', 'tumble'])




# set(['direly', 'finely', 'fireclay', 'firefly', 'firmly', 'firstly', 'fixedly', 'freely', 'liefly', 'refly', 'tiredly'])
