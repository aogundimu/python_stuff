#!/Applications/anaconda/bin/python


def map(len, data):
    result = []
    for ele in data:
        result.append(len(ele))

    return result
    
name_lengths = map(len, ["Mary", "Isla", "Sam"])

print (name_lengths)
