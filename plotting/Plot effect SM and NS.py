
import matplotlib.pyplot as plt
import sys
import numpy as np

if (len(sys.argv) == 1):
    print("Please specify an input file")
    sys.exit()

f = open(str(sys.argv[1]), "r")
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


#"Market_Size" = 0
#"Trust_in_Strangers" = 1
#"Signal_Reading" = 2
#"Trust_Rate" = 3
#"Cooperating_Agents" = 4
#"Trust_in_Neighbors" = 5
#"Trust_in_Newcomers" = 6

def unpackMeanDependenSocialMobility(index):
    rlist = [0 for i in range(n_different_mob_rates)]
    for i in range(n_different_mob_rates):
        for j in range(n_different_neighbourhood_sizes):
            rlist[i] += float(lines[j*n_different_mob_rates +1 + i][index])
    return [value / n_different_neighbourhood_sizes for value in rlist]

yAxis = [mob_rate for mob_rate in np.arange(mob_rate_min,mob_rate_max + 0.0001,mob_rate_stepsize)]
print(yAxis)
print(unpackMeanDependenSocialMobility(0))
#plt.title("Effect of social mobility with N = 1000 and n = {10..100}")
plt.plot(yAxis, unpackMeanDependenSocialMobility(0), label = "Market Size")
plt.plot(yAxis, unpackMeanDependenSocialMobility(1), label ="Trust in Strangers")
plt.plot(yAxis, unpackMeanDependenSocialMobility(2), label ="Signal Reading")
#plt.plot(yAxis, unpackMeanDependenSocialMobility(3), label ="Trust Rate")
#plt.plot(yAxis, unpackMeanDependenSocialMobility(4), label ="Cooperating_Agents")
#plt.plot(yAxis, unpackMeanDependenSocialMobility(5), label ="Trust in Neighbors")
#plt.plot(yAxis, unpackMeanDependenSocialMobility(6), label ="Trust in Newcomers")
plt.ylabel('Mean value')
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

yAxis = [ n for n in np.arange(n_min,n_max + 0.001, n_stepsize)]
#plt.title("Effect of neighborhood size with N = 1000 and  M = {0..1}")
plt.plot(yAxis, unpackMeanDependenNeighbourhood(2), label ="Signal Reading")
plt.plot(yAxis, unpackMeanDependenNeighbourhood(0), label = "Market Size")
plt.plot(yAxis, unpackMeanDependenNeighbourhood(1), label ="Trust in Strangers")
#plt.plot(yAxis, unpackMeanDependenNeighbourhood(3), label ="Trust Rate")
#plt.plot(yAxis, unpackMeanDependenNeighbourhood(4), label ="Cooperating_Agents")
#plt.plot(yAxis, unpackMeanDependenNeighbourhood(5), label ="Trust in Neighbors")
#plt.plot(yAxis, unpackMeanDependenNeighbourhood(6), label ="Trust in Newcomers")
plt.ylabel('Mean value')
plt.xlabel('Neighborhood Size')
plt.legend()
ax = plt.gca()
ax.set_ylim([0.3, 0.8])
plt.show()