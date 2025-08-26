import numpy as np

l=[1,2,3,4,5,6]
new_list = np.array_split(l,2)
print(new_list)
new_list =[list(x) for x in new_list]
print(new_list)