require 'initenv'
require 'cunn'
require'cutorch'
require 'nn'
plt = require 'gnuplot'
lim = 600
a1 = torch.Tensor(lim):zero()
b1 = torch.Tensor(lim):zero()
c1 = torch.Tensor(lim):zero()
a2 = torch.Tensor(lim):zero()
b2 = torch.Tensor(lim):zero()
c2 = torch.Tensor(lim):zero()


vanila_agent = torch.load('DQN3_0_1__FULL_Y_test_vanila_zork.t7') -- zork vanila
--vanila_agent = torch.load('DQN3_0_1__FULL_Y_largereplay3.t7') -- agent with the minimal amout of actions

for i=1,math.min(#vanila_agent.reward_history,lim) do
a1[i] = vanila_agent.reward_history[i]
--print(agent.obj_loss_history[i])
--loss = vanila_agent.obj_loss_history[i]
--b1[i] = loss[1]
--c1[i] = loss[2]
end
vanila_agent = nil

merged_agent = torch.load('DQN3_0_1__FULL_Y_test_obj_net_exploration_n_greedy_restriction_reval.t7')
for i=1,math.min(lim,#merged_agent.reward_history) do
a2[i] = merged_agent.reward_history[i]
--print(agent.obj_loss_history[i])
loss = merged_agent.obj_loss_history[i]
b2[i] = loss[1]
c2[i] = loss[2]
end
plt.figure(1)
plt.title('DQN Agent Reward')
plt.xlabel('epochs')
plt.plot({'Vanila Zork', a1:narrow(1,1,lim)},{'Action and Exploration Restriction', a2:narrow(1,1,lim)})
plt.movelegend('right','bottom')
plt.figure(2)
plt.plot({'Binary Cross Entropy loss', b2:narrow(1,1,lim)},{'Accuracy',c2:narrow(1,1,lim)})
plt.movelegend('right','bottom')
plt.title('Object Conv Net preformance')
plt.xlabel('epochs')
