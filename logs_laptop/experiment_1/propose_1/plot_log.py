import numpy as np
import matplotlib.pyplot as plt

with open('log.txt') as f:
    lines = [line for line in f]


log = []
count = 0
step_stack = []
reward_stack = []
for line in lines:
    count += 1
    # if count % 500 == 0:

    reads = line.split(',')
    reads = [cleanse.strip() for cleanse in reads]
    step_line = reads[1]
    reward_line = reads[3]
    # print(step_line)
    # print(reward_line)
    step_line = step_line.split(' ')
    step_num = int(step_line[2])
    # print(step_num)
    # print(step_num+1)
    reward_line = reward_line.split(' ')
    # print(reward_line)
    reward_num = float(reward_line[2])
    # print(reward_num)
    # print(reward_num+0.2)
    step_stack.append(step_num)
    reward_stack.append(reward_num)
    if count % 50 == 0:
        s = np.mean(step_stack)
        r = np.mean(reward_stack)
        log.append([s, r])
        step_stack = []
        reward_stack = []

    # if count > 5000:
    #     break

# print(log)
log  = np.array(log)
# print(log.shape)
t_log = np.transpose(log)
# print(t_log.shape)
# print(t_log)
x = t_log[0]
y = t_log[1]
# plt.plot(x, y)
# plt.scatter(t_log[0], t_log[1])
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('A3C Learning Log')
ax.set_xlabel('Frame')
ax.set_ylabel('Episodic Reward')
ax.set_ylim(-21, 21)
plt.show()