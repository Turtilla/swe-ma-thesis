import sys

data = sys.argv[1]
if len(sys.argv) > 2:
    new_data = sys.argv[2]
else:
    new_data = sys.argv[1]

def remove_ranges(tokens: list):
    '''A function that removes the "range" elements in a list based off of a conllu file. This is so as to
    exclude situations where in the list of tokens in a sentence one gets both "zrobiłem" and "zrobił" + "em".
    
    Args:
        tokens (list[str]): A list of token-tag pairs.

    Returns:
        A list of token-tag pairs with the elements without a tag (with "_" instead of it) are excluded.
    '''
    tokens = [x for x in tokens if ' _' not in x]
    return tokens

def remove_ranges_from_file(filename: str, newfile: str):
    '''A function that removes the "range" elements in a file based off of a conllu file. This is so as to
    exclude situations where in the list of tokens in a sentence one gets both "zrobiłem" and "zrobił" + "em".
    
    Args:
        filename (str): The name of the file to read from.
        newfile (str): The name of the file to write to.
    '''
    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines = remove_ranges(lines)
    
    with open(newfile, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    remove_ranges_from_file(data, new_data)