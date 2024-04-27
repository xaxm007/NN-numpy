#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# X1 = np.random.rand(5,3)
# np.random.seed(0)


# In[3]:


#X1 = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,1] ,[1,1,0]])
X1 = np.random.rand(5,3)
len(X1[0])


# In[4]:


print(X1)


# In[5]:


print(X1.shape[1])


# In[6]:


Y1 = np.array([[0] , [1] , [1] , [1] , [0]])


# In[7]:


print(Y1)


# In[8]:


#initialize weight and biases
def initialize_parameter(ip_size, hid_size, op_size):
    np.random.seed(0)
    w1 = np.random.randn(hid_size, ip_size) * 0.01
    b1 = np.zeros((hid_size,1))
    w2 = np.random.randn(op_size,hid_size) * 0.01
    b2 = np.zeros((op_size,1))
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return parameters

parameters = initialize_parameter(3, 4, 1)
print(parameters)


# In[9]:


# #sigmoid activation function
# def sigmoid(x):
#     return 1 / 1+np.exp(-x)
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# In[10]:


#forward propagation
def forward_prop(X1, parameters):
   #w1, b1, w2, b2 = parameters
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    
    z1 = np.dot(w1, X1.T) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    
    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2,cache
    
a2, cache = forward_prop(X1,parameters)


# In[11]:


# #binary cross-entropy loss function
# def binary_cross_entropy_loss(a2, Y1):
#     m = Y1.shape[0]
#     loss = -(1/m) * np.sum(Y1*np.log(a2) + (1-Y1)*np.log(1-a2))
#     return loss


# In[12]:


def binary_cross_entropy_loss(a2, Y1):
    m = Y1.shape[0]
    
#     Ensure a2 is within the valid range
    a2 = np.clip(a2, 1e-15, 1 - 1e-15)
    
    # Compute the loss
    loss = -(1/m) * np.sum(Y1*np.log(a2) + (1-Y1)*np.log(1-a2))
    
    return loss


# In[13]:


#backward propagation
def backward_prop(parameters, cache, X1, y):
    m = y.shape[0]
    
    z1 = cache["z1"]
    a1 = cache["a1"]
    z2 = cache["z2"]
    a2 = cache["a2"]
#     print(len(a2))
#     print(len(a2[0]))
    
#     da2 = - (y/a2) + ((1-y)/(1-a2))
#     dz2 = np.multiply(da2 , (a2 * (1-a2)))

# Ensure y and a2 have shape (1, 5)
    y = y.reshape(1, -1)
    a2 = a2.reshape(1, -1)
    
    # Calculate da2 with desired shape (1, 5)
    da2 = - (y / a2) + ((1 - y) / (1 - a2))
#     print(len(da2))
#     print(len(da2[0]))
#     dz2 = (da2 * (a2 * (1-a2)))
    dz2 = (a2 - y)





    
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    
    da1 = np.dot(parameters["w2"].T, dz2)
    dz1 = da1 * (a1 * (1-a1))
    
    dw1 = (1/m) * np.dot(dz1, X1)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    gradients = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    
    return gradients


# In[14]:


def update_parameters(parameters, gradients, learning_rate):
    dw1 = gradients["dw1"]
    dw2 = gradients["dw2"]
    db1 = gradients["db1"]
    db2 = gradients["db2"]
    
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    w1 = w1 - learning_rate*dw1
    b1 = b1 - learning_rate*db1
    w2 = w2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2
    
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2":b2}
    return parameters


# In[15]:


# def train(X, Y1, hidden_layer_size , num_iterations , learning_rate): 
#     parameters = initialize_parameter(X1.shape[1] , hidden_layer_size , 1)
#     for i in range(num_iterations): 
#         a2 , cache = forward_prop(X1 , parameters)
#         loss = binary_cross_entropy_loss(a2 , Y1)
#         gradients = backward_prop(parameters , cache , X1 , Y1)
#         parameters = update_parameters(parameters , gradients  , learning_rate)
#         if i% 1000 == 0 : 
#             print(f"iteration {i}: loss = {loss}")
#         return parameters 
# parameters = train(X1 , Y1 , hidden_layer_size = 4 , num_iterations = 10000 , learning_rate = 0.1 )


# In[16]:


# train the neural network
def train(X1, Y1, hidden_layer_size, num_iterations, learning_rate):
    # initialize the weights and biases
    parameters = initialize_parameter(X1.shape[1], hidden_layer_size, 1)
    
    for i in range(0, num_iterations):
        # forward propagation
        a2, cache = forward_prop(X1, parameters)
        
        # compute the loss
        loss = binary_cross_entropy_loss(a2, Y1)
        
        # backward propagation
        gradients = backward_prop(parameters, cache, X1, Y1)
        
        # update the parameters
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        if i % 1000 == 0:
            print(f"iteration {i}: loss = {loss}")
    
    return parameters

parameters = train(X1, Y1, hidden_layer_size=4, num_iterations=10000, learning_rate=0.001)


# In[18]:


get_ipython().system('pip install matplotlib')


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(-10, 10, 100)

# Compute the sigmoid functions
sigmoid_1 = 1 / (1 + np.exp(-x))
sigmoid_2 = np.exp(x) / (1 + np.exp(x))
sigmoid_3 = 1 / (1 + np.exp(x))

# Plot the sigmoid functions
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid_1, label=r'$\frac{1}{1 + e^{-x}}$', color='blue')
plt.plot(x, sigmoid_2, label=r'$\frac{e^{x}}{1 + e^{x}}$', color='green')
plt.plot(x, sigmoid_3, label=r'$\frac{1}{1 + e^{x}}$', color='red')

# Add labels and legend
plt.title('Graphs of Sigmoid Functions')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# In[ ]:




