
import matplotlib.pyplot as plt
import sys
import numpy as np

if (len(sys.argv) == 1):
    print("Please specify an input files")
    sys.exit()



#"Trust_in_Strangers" = 1


def unpackMeanDependenSocialMobility(index):
    rlist = [0 for i in range(n_different_mob_rates)]
    for i in range(n_different_mob_rates):
        for j in range(n_different_neighbourhood_sizes):
            rlist[i] += float(lines[j*n_different_mob_rates +1 + i][index])
    return [value / n_different_neighbourhood_sizes for value in rlist]

for file in sys.argv[1:]:
    f = open(str(file), "r")

    lines = f.read()
    lines = lines.split("\n")
    for i,line in enumerate(lines):
        lines[i] = line.split(" ")

    n_min = int(lines[0][0])
    n_max = int(lines[0][1])
    n_stepsize = int(lines[0][2])
    mob_rate_min = float(lines[0][3])
    mob_rate_max = float(lines[0][4])
    mob_rate_stepsize = float(lines[0][5])

    n_different_mob_rates = int((mob_rate_max - mob_rate_min + 0.0001)  /  mob_rate_stepsize +1)
    n_different_neighbourhood_sizes = int((n_max - n_min + 0.0001)  /  n_stepsize +1)
    yAxis = [mob_rate for mob_rate in np.arange(mob_rate_min,mob_rate_max + 0.0001,mob_rate_stepsize)]
    plt.plot(yAxis, unpackMeanDependenSocialMobility(1), label = file.split("Agent",1)[0] + " - Agent" )


plt.ylabel('Trust in strangers')
plt.xlabel('Social mobility')
plt.legend()
ax = plt.gca()
ax.set_ylim([0, 0.9])
plt.show()


def unpackMeanDependenNeighbourhood(index):
    rlist = [0 for i in range(n_different_neighbourhood_sizes)]
    for i in range(n_different_neighbourhood_sizes):
        for j in range(n_different_mob_rates):
            rlist[i] += float(lines[i*n_different_mob_rates +1 + j][index])
    return [value / n_different_mob_rates for value in rlist]

for file in sys.argv[1:]:
    f = open(str(file), "r")

    lines = f.read()
    lines = lines.split("\n")
    for i,line in enumerate(lines):
        lines[i] = line.split(" ")

    n_min = int(lines[0][0])
    n_max = int(lines[0][1])
    n_stepsize = int(lines[0][2])
    mob_rate_min = float(lines[0][3])
    mob_rate_max = float(lines[0][4])
    mob_rate_stepsize = float(lines[0][5])
    n_different_mob_rates = int((mob_rate_max - mob_rate_min + 0.0001)  /  mob_rate_stepsize +1)
    n_different_neighbourhood_sizes = int((n_max - n_min + 0.0001)  /  n_stepsize +1)
    yAxis = [ n for n in np.arange(n_min,n_max + 0.001, n_stepsize)]
    plt.plot(yAxis, unpackMeanDependenNeighbourhood(1), label = file.split("Agent",1)[0] + " - Agent")
plt.ylabel('Trust in strangers')
plt.xlabel('Neighbourhood size')
plt.legend()
ax = plt.gca()
#ax.set_ylim([0, 0.9])
plt.show()
