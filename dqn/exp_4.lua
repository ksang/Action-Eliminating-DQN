require 'plot'

local limit = 500
local exp_4_new_convQ_AEN = {}
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_111_lr0.00043_215a',limit)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_222_lr0.00043_215a',limit)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_333_lr0.00043_215a',limit)
plotExpFromTable(exp_4_new_convQ_AEN,"Average Cumulative Reward",nil,"Vanilla agents")

exp_4_new_convQ_AEN = {}
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_greedy_Q-arch_conv_Q_net_seed_111_AEN-arch_conv_AEN_max_10_sample_5_lr0.00043_215a',limit,nil,true)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_greedy_Q-arch_conv_Q_net_seed_333_AEN-arch_conv_AEN_max_10_sample_5_lr0.00043_215a',limit,nil,true)

plotExpFromTable(exp_4_new_convQ_AEN,"Average Cumulative Reward",nil,"Greedy agents")
