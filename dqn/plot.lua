require 'initenv'
require 'cunn'
require 'nn'
require 'gnuplot'

agent = torch.load('DQN3_0_1__FULL_Y_test_obj.t7')
a = torch.CudaTensor(800)
b = torch.CudaTensor(800)

for i=1,#agent.reward_history do 
a[i] = agent.reward_history[i]
b[i] = agent.obj_loss_history[i] end
gnuplot.figure(1)
gnuplot.plot({a:narrow(1,1,800)})
gnuplot.figure(2)
gnuplot.plot({b:narrow(1,1,800)})
