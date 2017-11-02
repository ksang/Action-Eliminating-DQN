require 'initenv'
require 'cunn'
require 'nn'
require 'gnuplot'

local lim = 800
local agent = torch.load('DQN3_0_1__FULL_Y_test_obj.t7')
local a = torch.CudaTensor(800):zero()
local b = torch.CudaTensor(800):zero()
local c = torch.CudaTensor(800):zero()
for i=1,#agent.reward_history do
a[i] = agent.reward_history[i]
--print(agent.obj_loss_history[i])
local loss = agent.obj_loss_history[i]
b[i] = loss[1]
c[i] = loss[2]
end
gnuplot.figure(1)
gnuplot.plot({a:narrow(1,1,lim)})
gnuplot.figure(2)
gnuplot.plot({b:narrow(1,1,lim)},{c:narrow(1,1,lim)})
