import matplotlib.pyplot as plt
import numpy as np

# -----Intilization---------
# State
x_raw = np.array([0, 1])
x = x_raw.reshape(-1,1)

x_hat_raw = np.array([0, 1.])
x_hat = x_hat_raw.reshape(-1,1)

# Cov
P = np.array([[1, 0], [0, 1]])

# Identity  Matrix
I = np.eye(2)

# State Transition Matrix
A= np.array([[1, 1], [0, 1]])

# Measurement Matrix
H = np.array([[1, 0], [0, 1]])

Q = np.array([[0.1, 0], [0, 0.1]])
R = np.array([[1, 0], [0, 1]])

# Set time step
step = 5

# ----The real------
# The real process noise
var_1 = np.sqrt(Q[0][0])
var_2 = np.sqrt(Q[1][1])

t = []
real_position = []
real_v = []

for i in range(step):
    t.append(i + 1)
    # Generate the process noise
    w_1 = np.random.normal(0, var_1, 1)
    w_2 = np.random.normal(0, var_2, 1)

    w_raw = np.array([w_1, w_2])
    w = w_raw.reshape(-1, 1)

    # Update the real state
    print("Step", i + 1, ":")
    print("Process Noise: ")
    print(w)
    print("Real State: ")
    x = A @ x + w 
    print(x)
    real_position.append(x[0])
    real_v.append(x[1])
    print("The real position list:", real_position)
    print("The real velocity list:", real_v)


# Draw Figures
plt.plot(t, real_position, label='Real Position', color='blue', marker='o')
plt.plot(t, real_v, label='Real velocity', color='red', marker='s')

plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('Real State')
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()