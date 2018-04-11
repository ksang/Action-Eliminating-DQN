require 'initenv'
require "Scale"
require 'NeuralQLearner'
require 'cutorch'
require 'nn'
require 'nnutils'
require 'cunn'
require 'nngraph'
require 'image'
fs = require 'paths'
plt = require 'gnuplot'
local a_r_table = {}
function addAgentToPlotTable(agent_name,title,limit,object_net_info, refresh)
   if not refresh and fs.filep(agent_name.."_result_summary.t7")  then
        addAgentFromSummary(agent_name,limit,title)
    else

      local agent = torch.load(agent_name..".t7")
      limit = math.min(#agent.reward_history,limit or #agent.reward_history)
      print (#agent.reward_history)
      DQN_reward = torch.Tensor(agent.reward_history)
      
      if agent.obj_loss_history and object_net_info then
        local AEN_loss ,AEN_acc = torch.zeros(limit),torch.zeros(limit)
        for i=1,limit do
          AEN_loss[i],AEN_acc[i] = agent.obj_loss_history[i][1], agent.obj_loss_history[i][2]
        end
      end
      table.insert(a_r_table,{  title or string.gsub(agent_name, "_", " " ) , DQN_reward:narrow(1,1,math.min(DQN_reward:size()[1],limit))})
      if object_net_info then
        local AEN_stat_table ={{'Binary Cross Entropy loss', AEN_loss:narrow(1,1,limit)},{'Accuracy',AEN_acc:narrow(1,1,limit)}}
        --plot seperate graph for object network
        plotExpFromTable(AEN_stat_table,nil,nil,title or agent_name .. " AEN Preformanc",nil,nil)
      end
      torch.save(agent_name.."_result_summary.t7",{reward = DQN_reward,loss = AEN_loss,acc = AEN_acc,limit =limit, title =title or agent_name})
    end
end

function addAgentFromSummary(agent_name,limit,title)
    local agent_summary = torch.load(agent_name.."_result_summary.t7")
    limit = limit or agent_summary.limit
    title = title or string.gsub(agent_name, "_", " " )
    if agent_summary.loss then
      plt.figure()
      plt.title(title.." Object Network Preformance")
      plt.plot({'Binary Cross Entropy loss', agent_summary.loss:narrow(1,1,math.min(limit,agent_summary.limit))},{'Accuracy',agent_summary.acc:narrow(1,1,limit)})
      plt.movelegend('right','bottom')
      plt.xlabel('Epochs, 10000 steps per epoch')
    end
    print(agent_summary.reward)
	table.insert(a_r_table,{title, torch.Tensor(agent_summary.reward):narrow(1,1,limit)})
end
--gif should be a string containing png name
function plotExpFromTable(table,ylabel,legend_pos,title,fig_num,png)
    if png ~= nil then 
      plt.pngfigure(png)
    else 
      plt.figure(fig_num)
    end 
    --plt.title('title')
    plt.xlabel('Steps 10k')
    plt.ylabel(ylabel)
    legend_pos = legend_pos or {'right','bottom'}
    plt.movelegend(unpack(legend_pos))
    plt.plot(a_r_table)
    gnuplot.plotflush()
end

--insert agents here
--EGG 5  take actions
--[[
local limit= 190
addAgentFromSummary("DQN3_0_1__FULL_Y_test_zork_vanila_1mil_replay",limit,"Vanilla")
addAgentToPlotTable( "DQN3_0_1__FC_restrict_exploration", "AE-Explore", limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_1_lr1.7e7","AE-Greedy",limit,false)
addAgentToPlotTable( "DQN3_0_1__FC_restrict_exploration_n_action", "AE-DQN", limit,false)
plotExpFromTable(a_r_table,nil,nil,nil,"iclr/egg-5obj-iclr.png")
--plt.pngfigure("iclr/egg-5obj-iclr.png")
--plt.title('DQN Agent Reward - limited action space')
--plt.xlabel('Steps 10k')
--plt.ylabel('Average Cumulative Reward')
--plt.movelegend('right','bottom')
--plt.plot(a_r_table)
--gnuplot.plotflush()
]]
--[[
--EGG 30 take actions
a_r_table = {}
addAgentToPlotTable("DQN3_0_1__zork_FC_vanila_scenario_2_lr_1.9e7","Vanilla",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_2_lr1.7e7","Greedy",limit,false)
addAgentToPlotTable("DQN3_0_1__zork_FC_merged_scenario_2_lr_1.9e7","Merged",limit,false)
--plot in main reward graph
plt.figure()
plt.title('DQN Agent Reward - extended action space')
plt.xlabel('Epochs, 10000 steps per epoch')
plt.movelegend('right','bottom')
plt.plot(a_r_table)
]]
--EGG 200 take actions
--[[
a_r_table = {}
addAgentToPlotTable("DQN3_0_1_zork_FC_vanila_scenario_3_lr1.7e7_200a","Vanilla",limit,false)
--addAgentToPlotTable("DQN3_0_1_zork_FC_merged_scenario_3_lr1.7e7","Merged 1.7e7",limit,true)
addAgentToPlotTable("DQN3_0_1_zork_FC_explore_amended_scenario_3_max_3_sample_5_drop_prob_0.9_lr1.7e-06_209a","AE-Explore",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_3_lr1.7e-06_209a","AE-Greedy",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_merged_scenario_3_lr1.7e7_200a","AE-DQN",limit,false)

plt.pngfigure("iclr/egg-200obj-iclr.png")
--plt.title('DQN Agent Reward - extreme action space')
plt.xlabel('Steps 10k')
plt.ylabel('Average Cumulative Reward')
plt.movelegend('right','bottom')
plt.plot(a_r_table)
gnuplot.plotflush()

--Troll 200 take actions
a_r_table = {}
limit = 500
addAgentToPlotTable("DQN3_0_1_zork_FC_vanila_scenario_4_lr1.7e-06_215a","Vanilla",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_explore_amended_scenario_4_max_3_sample_5_drop_prob_0.9_lr1.7e-06_215a","AE-Explore",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_4_lr1.7e-06_215a","AE-Greedy",limit,false)
--addAgentToPlotTable("DQN3_0_1_zork_FC_merged_amended_scenario_4_max_5_sample_5_drop_prob_0.9_lr1.7e-06_215a","Merged 1.7e-6",limit,true,true)
addAgentToPlotTable("BACK/DQN3_0_1_zork_FC_merged_scenario_4_lr1.7e-06_215a","AE-DQN",limit,false)

plt.pngfigure('iclr/troll-1-200obj-iclr.png')
--plt.title('DQN Agent Reward: Troll quest ,-1 step, extreme action space')
plt.xlabel('Steps 10k')
plt.ylabel('Average Cumulative Reward')
plt.movelegend('left','top')
plt.plot(a_r_table)


--double step penalty Troll quest
--[[
a_r_table = {}
limit = 990
addAgentToPlotTable("DQN3_0_1_zork_FC_vanila_scenario_4_step_-2_lr1.7e-06_215a","Vanilla 1.7e-6",limit,false)
--addAgentToPlotTable("DQN3_0_1_a_r_table = {}
zork_FC_merged_amended_scenario_4_step_-2_sample_5_drop_prob_0.8_lr1.7e-06_215a", "Merged s5 1.7e-6",limit,true)
addAgentToPlotTable("DQN3_0_1_zork_FC_merged_amended_scenario_4_step_-2_sample_10_drop_prob_0.99_lr1.7e-06_215a","Merged s10 1.7e-6",limit)
plt.figure(4)
plt.title('DQN Agent Reward: Troll quest, -2 step, extreme action space')
plt.xlabel('Epochs, 10000 steps per epoch')
plt.movelegend('right','bottom')
plt.plot(a_r_table)
a_r_table = {}
gnuplot.plotflush()
]]
--experiments 
--limit = 95
--a_r_table = {}
--addAgentToPlotTable("zork_scenario_1_merged_Q-arch_conv_q_net_AEN-arch_conv_obj_net_max_2_sample_1_drop_prob_0.9_lr0.0025_14a","2 conv merged lr 0.0025",limit,true)
--addAgentToPlotTable("zork_scenario_1_merged_Q-arch_1HNN100_AEN-arch_linear_max_2_sample_1_drop_prob_0.9_lr0.0017_14a","hnn + lin merged lr 0.0017",limit,true)
--plotExpFromTable(a_r_table,"Average Cumulative Reward",nil,"egg 5",nil,nil)

limit = 195
a_r_table = {}
addAgentToPlotTable("zork_scenario_3_vanila_Q-arch_conv_q_net_lr0.0025_209a",nil,limit)
--addAgentToPlotTable("zork_scenario_3_merged_Q-arch_1HNN100_AEN-arch_linear_max_2_sample_1_drop_prob_0.9_lr0.0017_209a",nil,limit,true)
--addAgentToPlotTable("zork_scenario_3_vanila_Q-arch_1HNN100_lr0.0017_209a",nil,limit,false,true)
addAgentFromSummary("zork_scenario_3_merged_Q-arch_conv_q_net_AEN-arch_conv_obj_net_max_5_sample_5_drop_prob_0.9_lr0.0025_209a",limit)
addAgentFromSummary("zork_scenario_3_greedy_Q-arch_conv_q_net_AEN-arch_conv_obj_net_max_5_sample_5_lr0.0025_209a",limit)
plotExpFromTable(a_r_table,"Average Cumulative Reward",nil,"egg 200",nil,nil)
