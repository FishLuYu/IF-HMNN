import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'

train_accuracy = history1.history['accuracy']
val_accuracy = history1.history['val_accuracy']
train_loss = history1.history['loss']
val_loss = history1.history['val_loss']

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(train_accuracy, label='train_accuracy', linewidth=2, color='navy')
ax1.plot(val_accuracy, label='val_accuracy', linewidth=2, color='crimson')
ax1.legend(fontsize=10, loc='lower right')
ax1.set_xlabel('Epoch', fontsize=12, fontname='Times New Roman')
ax1.set_ylabel('Accuracy', fontsize=12, fontname='Times New Roman')
ax1.set_title('Train & Validation Accuracy', fontsize=14, fontname='Times New Roman')
ax1.tick_params(axis='both', which='major', labelsize=10)

# ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(train_loss, label='train_loss', linewidth=2, color='navy')
# ax2.plot(val_loss, label='val_loss', linewidth=2, color='crimson')
# ax2.legend(fontsize=10, loc='upper right')
# ax2.set_xlabel('Epoch', fontsize=12, fontname='Times New Roman')
# ax2.set_ylabel('Loss', fontsize=12, fontname='Times New Roman')
# ax2.set_title('Train & Validation Loss', fontsize=14, fontname='Times New Roman')
# ax2.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()

plt.show()