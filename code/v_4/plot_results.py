import matplotlib.pyplot as plt
# Read the loss values from the file
loss_values = []
with open('training_log.txt', 'r') as file:
    for line in file:
        loss_values.append(float(line.strip()))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('training_loss_plot.png', bbox_inches='tight')
