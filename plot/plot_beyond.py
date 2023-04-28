import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
import matplotlib

speedup_a = np.zeros((5, 9))
acc_a = np.zeros((5, 9))


f_snicit_a = open("../log/beyond/snicit_a.txt")
lines = f_snicit_a.readlines()

dict_B_a = {1000:0, 2000:1, 2500:2, 5000:3, 10000:4}

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    speedup_a[dict_B_a[int(s[1])]][int(s[0]/2)] = s[3]
    acc_a[dict_B_a[int(s[1])]][int(s[0]/2)] = s[2]

f_snicit_a.close()

f_snig_a = open("../log/beyond/snig_a.txt")
lines = f_snig_a.readlines()

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    for t in range(0, 18, 2):
        speedup_a[dict_B_a[int(s[0])]][int(t/2)] = s[2] / speedup_a[dict_B_a[int(s[0])]][int(t/2)]
        acc_a[dict_B_a[int(s[0])]][int(t/2)] = s[1] - acc_a[dict_B_a[int(s[0])]][int(t/2)]

f_snig_a.close()


speedup_b = np.zeros((5, 9))
acc_b = np.zeros((5, 9))


f_snicit_b = open("../log/beyond/snicit_b.txt")
lines = f_snicit_b.readlines()

dict_B_b = {1000:0, 2000:1, 2500:2, 5000:3, 10000:4}

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    speedup_b[dict_B_b[int(s[1])]][int(s[0]/2)] = s[3]
    acc_b[dict_B_b[int(s[1])]][int(s[0]/2)] = s[2]

f_snicit_b.close()

f_snig_b = open("../log/beyond/snig_b.txt")
lines = f_snig_b.readlines()

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    for t in range(0, 18, 2):
        speedup_b[dict_B_b[int(s[0])]][int(t/2)] = s[2] / speedup_b[dict_B_b[int(s[0])]][int(t/2)]
        acc_b[dict_B_b[int(s[0])]][int(t/2)] = s[1] - acc_b[dict_B_b[int(s[0])]][int(t/2)]

f_snig_b.close()


speedup_c = np.zeros((5, 6))
acc_c = np.zeros((5, 6))


f_snicit_c = open("../log/beyond/snicit_c.txt")
lines = f_snicit_c.readlines()

dict_B_c = {1000:0, 2000:1, 2500:2, 5000:3, 10000:4}

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    speedup_c[dict_B_c[int(s[1])]][int(s[0]/2)] = s[3]
    acc_c[dict_B_c[int(s[1])]][int(s[0]/2)] = s[2]

f_snicit_c.close()

f_snig_c = open("../log/beyond/snig_c.txt")
lines = f_snig_c.readlines()

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    for t in range(0, 12, 2):
        speedup_c[dict_B_c[int(s[0])]][int(t/2)] = s[2] / speedup_c[dict_B_c[int(s[0])]][int(t/2)]
        acc_c[dict_B_c[int(s[0])]][int(t/2)] = s[1] - acc_c[dict_B_c[int(s[0])]][int(t/2)]

f_snig_c.close()

speedup_d = np.zeros((5, 6))
acc_d = np.zeros((5, 6))


f_snicit_d = open("../log/beyond/snicit_d.txt")
lines = f_snicit_d.readlines()

dict_B_d = {1000:0, 2000:1, 2500:2, 5000:3, 10000:4}

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    speedup_d[dict_B_d[int(s[1])]][int(s[0]/2)] = s[3]
    acc_d[dict_B_d[int(s[1])]][int(s[0]/2)] = s[2]

f_snicit_d.close()

f_snig_d = open("../log/beyond/snig_d.txt")
lines = f_snig_d.readlines()

for line_idx in range(len(lines)):
    if line_idx == 0:
        continue
    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', lines[line_idx])]
    for t in range(0, 12, 2):
        speedup_d[dict_B_d[int(s[0])]][int(t/2)] = s[2] / speedup_d[dict_B_d[int(s[0])]][int(t/2)]
        acc_d[dict_B_d[int(s[0])]][int(t/2)] = s[1] - acc_d[dict_B_d[int(s[0])]][int(t/2)]

f_snig_d.close()


matplotlib.rcParams.update({'font.size': 20})
cmap1 = plt.get_cmap(sns.color_palette("ch:start=2,rot=0.1", as_cmap=True)).copy()
cmap1.set_under('none')
cmap3 = plt.get_cmap(sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)).copy()
cmap3.set_over('none')

# Labels
xlabs = ["0", "2", "4", "6", "8", "10", "12", "14", "16"]
ylabs = ["1000", "2000", "2500", "5000", "10000"]
         
# Heat map
fig, ax = plt.subplots()

ax1 = sns.heatmap(speedup_a, cmap=cmap1, cbar_kws={'pad': 0.02, "shrink": .48}, xticklabels=xlabs, yticklabels=ylabs,square=True)
plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.tight_layout()
plt.savefig("figs/speedup-a.jpg", dpi=1800)

plt.clf()

ax1 = sns.heatmap(acc_a, cmap=cmap3, cbar_kws={'pad': 0.02, "shrink": .48}, xticklabels=xlabs, yticklabels=ylabs,square=True)
plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/acc-a.jpg", dpi=1800)

plt.clf()


ax2 = sns.heatmap(speedup_b, cmap=cmap1, cbar_kws={'pad': 0.02, "shrink": .48}, xticklabels=xlabs, yticklabels=ylabs,square=True)

plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/speedup-b.jpg", dpi=1800)

plt.clf()

ax2 = sns.heatmap(acc_b, cmap=cmap3, cbar_kws={'pad': 0.02, "shrink": .48}, xticklabels=xlabs, yticklabels=ylabs,square=True)
plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/acc-b.jpg", dpi=1800)

plt.clf()


xlabs = ["0", "2", "4", "6", "8", "10"]
ax3 = sns.heatmap(speedup_c, cmap=cmap1, cbar_kws={'pad': 0.02, "shrink": .72}, xticklabels=xlabs, yticklabels=ylabs,square=True)

plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/speedup-c.jpg", dpi=1800)

plt.clf()

ax3 = sns.heatmap(acc_c, cmap=cmap3, cbar_kws={'pad': 0.02, "shrink": .72}, 
                    xticklabels=xlabs, yticklabels=ylabs,square=True)
plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/acc-c.jpg", dpi=1800)

plt.clf()

ax4 = sns.heatmap(speedup_d, cmap=cmap1, cbar_kws={'pad': 0.02, "shrink": .72}, xticklabels=xlabs, yticklabels=ylabs,square=True)

plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/speedup-d.jpg", dpi=1800)

plt.clf()

ax4 = sns.heatmap(acc_d, cmap=cmap3, cbar_kws={'pad': 0.02, "shrink": .72}, 
                    xticklabels=xlabs, yticklabels=ylabs,square=True)
plt.xlabel(r"$t$")
plt.ylabel(r"$B$")
plt.savefig("figs/acc-d.jpg", dpi=1800)

plt.clf()