import json
import numpy as np
import matplotlib.pyplot as plt
from util.misc import KL
import pickle

# set width of bar 
barWidth = 0.2
fig = plt.subplots(figsize =(12, 8)) 

# Set position of bar on X axis 
br1 = np.arange(3) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3] 

alpha_num = [1, 0.5, 0.3]
xlabel_pos = [x + barWidth/2 for x in br2]
xlabel_name = [r"$\alpha_{test}$ = " + f"{alpha_num[alpha_idx]}" for alpha_idx in np.arange(3)]

policy_label = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS"]

# MNIST

# shift_fedavg = [0.5101901916, 0.6230495868, 0.8451683918]
# err_fedavg = [0.1250681747, 0.2891163942, 0.2666703559]/np.sqrt(5)

# shift_poc = [0.5462785024, 0.7515671883, 0.8153331924]
# err_poc = [0.1395014707, 0.2758284238, 0.2553444897]/np.sqrt(5)

# shift_fedcbs = [0.4969238078, 0.6859256702, 0.8354198855]
# err_fedcbs = [0.1304184017, 0.204644962, 0.2595669758]

# shift_fedqs = [0.2055645164, 0.2235374998, 0.2047348319]
# err_fedqs = [0.07382775928, 0.1160149153, 0.1052931974]/np.sqrt(5)

# CIFAR 10
# shift_fedavg = [0.3, 0.48, 0.72]
# err_fedavg = [0.08, 0.15, 0.16]/np.sqrt(5)

# shift_poc = [0.31, 0.48, 0.69]
# err_poc = [0.09, 0.14, 0.14]/np.sqrt(5)

# shift_fedcbs = [0.29, 0.49, 0.66]
# err_fedcbs = [0.12, 0.16, 0.17]/np.sqrt(5)

# shift_fedqs = [0.03, 0.08, 0.16]
# err_fedqs = [0.02, 0.06, 0.06]/np.sqrt(5)

# Fashion-MNIST
shift_fedavg = [0.5, 0.61, 0.74]
err_fedavg = [0.13, 0.23, 0.07]/np.sqrt(5)

shift_poc = [0.61, 0.75, 0.77]
err_poc = [0.10, 0.21, 0.13]/np.sqrt(5)

shift_fedcbs = [0.50, 0.57, 0.66]
err_fedcbs = [0.12, 0.19, 0.07]/np.sqrt(5)

shift_fedqs = [0.09, 0.13, 0.16]
err_fedqs = [0.05, 0.07, 0.05]/np.sqrt(5)

plt.bar(br1, shift_fedavg, width = barWidth, color='b', edgecolor ='none', label ='FedAvg')
plt.bar(br2, shift_poc, width = barWidth, color='r', edgecolor ='none', label ='POWER-OF-CHOICE')
plt.bar(br3, shift_fedcbs, width = barWidth, color ='y', edgecolor ='none', label ='Fed-CBS')
plt.bar(br4, shift_fedqs, width = barWidth, color = 'g', edgecolor ='none', label ='FedQS')

plt.errorbar(br1, shift_fedavg, yerr= err_fedavg, fmt='ko', capsize=5)
plt.errorbar(br2, shift_poc, yerr= err_poc, fmt='ko', capsize=5)
plt.errorbar(br3, shift_fedcbs, yerr= err_fedcbs, fmt='ko', capsize=5)
plt.errorbar(br4, shift_fedqs, yerr= err_fedqs, fmt='ko', capsize=5)

plt.xticks(xlabel_pos, xlabel_name, fontsize=20)
plt.legend(fontsize=15)
plt.ylabel(r"$D(\bar{S}(T))$", rotation=90, fontsize=20)
# plt.errorbar(policy_label, shift_alpha1, 
#              yerr = error_alpha1, 
#              fmt ='o')
plt.show()