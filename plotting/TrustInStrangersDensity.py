
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
    #data = data.replace(" ","")
    retList = []

    for x in data.split(" "):
        if x == "":
            continue
        if x == "nan":
            retList.append(float(0))
            continue
        try:
            retList.append(float(x))
        except ValueError:
            retList.append(float(0))
            continue
    return retList #[float(x) for x in data.split(" ") if x != ""]

fig, (ax1, ax2, ax3, ax4, ax5,ax6,ax7,ax8,ax9,ax10,ax11) = plt.subplots(11, sharex=True, sharey= True)
axis = [ax1, ax2, ax3, ax4, ax5,ax6,ax7,ax8,ax9,ax10,ax11]
for i,mob_rate in enumerate(np.arange(mob_rate_min,mob_rate_max + 0.0001,mob_rate_stepsize)):
    density = gaussian_kde(lineToList(lines[i+1]))
    xs = np.linspace(0,1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    axis[10-i].plot(xs,density(xs))
    axis[10-i].text(0, 1.5, "mobility = " + (str)(round(mob_rate,3)), fontsize = 9)
    axis[10-i].yaxis.set_visible(False)
    

plt.text(-0.1, 30, "Density", fontsize = 11, rotation=90)
plt.xlabel('Trust in Strangers')
ax = plt.gca()
plt.show()
