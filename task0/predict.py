testset = open("test.csv").readlines()[1:]
output = open("submission.csv", "w")
output.write("Id,y\n")

avg = lambda lst: sum([float(x) for x in lst])/len(lst)

for line in testset:
    l = line.split(',')
    output.write("{},{}\n".format(l[0], avg(l[1:])))

output.close()
