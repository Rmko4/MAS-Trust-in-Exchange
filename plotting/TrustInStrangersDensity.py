
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import gaussian_kde

if (len(sys.argv) == 1):
    print("Please specify an input file")
    sys.exit()

f = open(str(sys.argv[1]), "r")
lines = f.read()
lines = lines.split("\n")

firstLine = lines[0].split(" ")
n_min = int(firstLine[0])
n_max = int(firstLine[1])
n_stepsize = int(firstLine[2])
mob_rate_min = float(firstLine[3])
mob_rate_max = float(firstLine[4])
mob_rate_stepsize = float(firstLine[5])


def lineToList(data):
    data = data.replace("[","")
    data = data.replace("]","")
    data = data.replace("'","")
    data = data.replace(" ","")
    return [float(x) for x in data.split(",")]

for i,mob_rate in enumerate(np.arange(mob_rate_min,mob_rate_max + 0.0001,mob_rate_stepsize)):
    density = gaussian_kde(lineToList(lines[i+1]))
    xs = np.linspace(0,1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),label =" social mobility = " + (str)(round(mob_rate,3)))

plt.ylabel('Density')
plt.xlabel('Trust in Strangers')
plt.legend()
ax = plt.gca()
plt.show()
