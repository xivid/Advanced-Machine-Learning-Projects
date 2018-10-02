import numpy as np
testset = open("test.csv").readlines()[1:]
testset = [s.strip() for s in testset]
output = open("submission.csv", "w")
output.write("Id,y\n")

savg = lambda l: sum([float(x) for x in l]) / len(l)

def avg(l):
    a = []
    min0 = 11111
    for x in l:
        x = x.split('.')
        if len(x[1]) > 20:
            print("error " + x[0] + '.' + x[1])
        y = int(x[0] + x[1])*(10**(20-len(x[1])))
        min0 = min(min0, 20 - len(x[1]))
        a.append(y)
    s = str(sum(a))
    return s[:-21]+'.'+s[-21:-min0]

for line in testset:
    l = line.split(',')
#    output.write("{},{},{}\n".format(l[0], avg(l[1:]), savg(l[1:])))
    output.write("{},{}\n".format(l[0], avg(l[1:])))

output.close()
