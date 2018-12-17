#!/Applications/anaconda/bin/python

def recursiveMult(num1, num2):
    

### This assumes that the length of the number will be a power of 2
### Also that both numbers are the same length
### This also assumes that the numbers are in a string form
def karatsuba(num1, num2):

    length = len(num1)
    lh = int(length/2)
    
    a = num1[0:lh]
    b = num1[lh:]
    c = num2[0:lh]
    d = num2[lh:]
    
    an = int(a)
    bn = int(b)
    cn = int(c)
    dn = int(d)
    
    # step 1
    ac = an * cn
    # print( "step 1 = ", ac)
    
    # step 2
    bd = bn * dn
    #print( "step 2 = ", bd )
    
    # step 3
    abcd = (an + bn) * (cn + dn)
    # print("step 3 = ", abcd )
    
    # step 4
    diff = abcd - bd - ac
    #print("step 4 = ", diff )

    result_1 = int(ac * (10 ** length))
    # print( result_1)

    result_2 = int(bd)
    # print( result_2 )

    result_3 = int( diff * 10**(lh) )
    # print( result_3 )
    
    result = ( result_1 + result_2 + result_3 )

    return result
    
########################################



n1 = '3141592653589793238462643383279502884197169399375105820974944592'

# a = 31415926535897932384626433832795
# b = 02884197169399375105820974944592
 

n2 = '2718281828459045235360287471352662497757247093699959574966967627'
# print( len(n1) )
# print( len(n2) )

# print( int( len(n1) / 2) )
#print( karatsuba('5678', '1234') )

print( karatsuba(n1,n2) )
