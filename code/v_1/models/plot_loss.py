import matplotlib.pyplot as plt

# Read training losses from the file
with open('training_log.txt', 'r') as file:
    train_losses = [float(line.strip()) for line in file.readlines()]

# Plotting the training loss
#plt.figure(figsize=(10, 5))
#print(train_losses)
plt.plot(train_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.show()

