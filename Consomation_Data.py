import numpy as np
conso_file = open(r"RES2-6-9.csv")
file_data = conso_file.readlines()[1:]
DATA=[]
for i in file_data:
    DATA.append(i)

print(len(DATA))