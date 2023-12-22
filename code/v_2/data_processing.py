#code for computing auxillary variables


# import necessary libraries
import numpy as np



# process dataset to remove nan s 
# Load data from npy files
X_neural = np.load('X_neural.npy')
Y_target = np.load('Y_target.npy')
# replace all nan values by 1 in neural spike observations.
X_neural[np.isnan(X_neural)] = 1
# replace all nan values by 0 in observation data.
Y_target[np.isnan(Y_target)] = 0
print("Checking nan in dataset")
print(np.max(X_neural))

Z = np.zeros((1,X_neural.shape[1]))
i = 0
for i in range(X_neural.shape[0]):
    Z_i = 0.5*Z[i,:] + 0.5*X_neural[i,:]
    Z = np.vstack([Z,Z_i])
    if i%1000 == 0:
        print(i)
    if i < 100:
        print(Z)
print("Completed computing auxillary variables")
print(Z)

