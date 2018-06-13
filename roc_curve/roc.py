from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

actual = []
predictions = []

for i in range (0,10000):
    if i < 9765:
        actual.append(0)
        predictions.append(0)
    else:
        actual.append(1)
        predictions.append(2)

print actual 
print '################'
print predictions 

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
# print false_positive_rate
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()