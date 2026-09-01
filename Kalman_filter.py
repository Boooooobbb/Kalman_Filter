import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)  
# -----Initialization---------
# Set time step
step = 40
""" while True:
    try:
        step = int(input('请输入时间步（正整数）：'))
        if step <= 0:
            print('请输入正整数！')
            continue
        break
    except ValueError:
        print('输入无效，请输入一个整数。') """

# Initialize the real state
x_raw = np.array([0, 1])
x = x_raw.reshape(-1,1)

# Initialize the posterior state
x_pos_raw = np.array([0, 1.])
x_pos = x_pos_raw.reshape(-1,1)

# Initialize the covariance of posterior prediction
p = np.array([[1, 0], [0, 1]])

# Identity  Matrix
I = np.eye(2)

# State Transition Matrix
A= np.array([[1, 1], [0, 1]])

# Measurement Matrix
H = np.array([[1, 0], [0, 1]])
 
# Mean
mean = np.zeros(2)

# Covarience Matrix of process noise
Q = np.array([[0.1, 0], [0, 0.1]])

# Covariance Matrix of measurement noise
R = np.array([[1, 0], [0, 1]])

# The real process list
t = []
real_position = []
real_v = []

# The measurement list
mea_position = []
mea_v = []

# The prior prediction list
prior_position = []
prior_v = []

# The posterior prediction list
pos_position = []
pos_v = []

for i in range(step):
    t.append(i + 1)
    # Generate the process noise
    w = np.random.multivariate_normal(mean, Q).reshape(2, 1)

    # Generate the measurement noise
    v = np.random.multivariate_normal(mean, R).reshape(2, 1)

    # Update the real state
    x = A @ x + w 
    real_position.append(x[0])
    real_v.append(x[1])

    # Update the measurement
    z = H @ x + v
    mea_position.append(z[0])
    mea_v.append(z[1])

    # The Prior Prediction
    x_prior = A @ x_pos
    p_prior = A @ p @ A.T + Q

    prior_position.append(x_prior[0])
    prior_v.append(x_prior[1])

    # The Posterior Prediction
    S = H @ p_prior @ H.T + R
    K = p_prior @ H.T @ np.linalg.inv(S)
    x_pos = x_prior + K @ (z - H @ x_prior)

    pos_position.append(x_pos[0])
    pos_v.append(x_pos[1])

    # Update p
    p = (I - K @ H) @ p_prior

    print("---Step", i + 1, "---")
    print("Process Noise: ")
    print(w)
    print("Real State: ")
    print(x)
    print("The measurement noise:")
    print(v)
    print("Measurement:")
    print(z)
    print("Prior Prediction:")
    print(x_prior)

# ========== Draw Figures ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2行1列，共享x轴

# ---------- Position ----------
ax1.plot(t, real_position, label='Real Position', color='blue', marker='o', markersize=4)
ax1.plot(t, mea_position, label='Measured Position', color='green', marker=',', linestyle='--')
ax1.plot(t, prior_position, label='Prior Position', color='red', marker='s', markersize=4)
ax1.plot(t, pos_position, label='Posterior Position', color='orange', marker='.', markersize=4)
ax1.set_ylabel('Position')
ax1.set_title('Kalman Filter - Position & Velocity')
ax1.legend(loc='best')
ax1.grid(True)

# ---------- Velocity ----------
ax2.plot(t, real_v, label='Real Velocity', color='blue', marker='o', markersize=4)
ax2.plot(t, mea_v, label='Measured Velocity', color='green', marker=',', linestyle='--')
ax2.plot(t, prior_v, label='Prior Velocity', color='red', marker='s', markersize=4)
ax2.plot(t, pos_v, label='Posterior Velocity', color='orange', marker='.', markersize=4)
ax2.set_xlabel('Time')
ax2.set_ylabel('Velocity')
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
plt.show()