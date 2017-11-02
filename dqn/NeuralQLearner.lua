--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'optim'

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #args.actions
    --print(table.unpack(self.actions))
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.ncols,args.state_rows, self.hist_len*args.state_cols}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512
    self.Y_buff         = torch.CudaTensor(self.minibatch_size,5) -- buffer for multiclass labels
    self.valid_Y_buff   = torch.CudaTensor(self.valid_size,5)
    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end
-- init obj network
    self.obj_network = 'conv_obj_net'
    local msg, err = pcall(require, self.obj_network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.obj_network)
        if not err_msg then
            error("Could not find object network file ")
        end
        if self.best and exp.best_model then
            self.obj_network = exp.best_model
        else
            self.obj_network = exp.model
        end
    else
        print('Creating Object Network from ' .. self.obj_network)
        self.obj_network = err
        self.obj_network = self:obj_network()

    end
    -- set object network loss for multi lable learning
    self.objNetLoss = nn.MultiLabelSoftMarginCriterion()
    --self.objNetLoss = nn.BCECriterion()
    self.optimState = {learningRate = 0.000013}--,learningRateDecay = 0.01}--, nesterov = true, momentum = 0.8, dampening = 0} -- for obj network

    -- end of object network init

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()

    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.obj_network:cuda()
        self.Y_buff:cuda()
	      self.valid_Y_buff:cuda()
        self.objNetLoss:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.obj_network:float()
        self.tensor_type = torch.FloatTensor

    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.lastAction_o = nil
    self.lastAction_bad = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.obj_w, self.obj_dw = self.obj_network:getParameters()
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    print("number of parameterns in object network",#self.obj_w)
    print("number of parameterns in state network",#self.w)


    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        --print("@debug preprocess",self.preproc:forward(rawstate:float()):clone():reshape(self.state_dim))

        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end
    --print("@debug no preprocess",rawstate)
    return rawstate

end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end
    --print(s2:size())
    -- Compute max_a Q(s_2, a).
    q2_max = target_q_net:forward(s2):float():max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, s_for_obj, a_for_obj,a_o, bad_command = self.transitions:sample(self.minibatch_size)
    --print("@DEBUG - transition batch sample", torch.cat(torch.cat(a ,a_o,2), bad_command,2))
    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}
    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
    self:objLearnMiniBatch(s_for_obj,a_for_obj,a_o, bad_command) -- now train obj network
end

function nql:setYbuff(action_object, bad_command, validation)

  --@FIXME this is ugly, cant seem to change the needed buffer when using the buffer as argument
   if validation then
   self.valid_Y_buff = self.valid_Y_buff:zero() + 0.5 -- 2d buffer of size minibatchXnum_objects set 0.5 to unknown entries
   --print("before",buff)
   local object_index
   for i=1, (#self.valid_Y_buff)[1] do -- expand labels
      object_index = action_object[i]
      self.valid_Y_buff[i][object_index] = bad_command[i]
   end

   else
     self.Y_buff = self.Y_buff:zero() + 0.5 -- 2d buffer of size minibatchXnum_objects set 0.5 to unknown entries
     --print("before",buff)
     local object_index
     for i=1, (#self.Y_buff)[1] do -- expand labels
        object_index = action_object[i]
        self.Y_buff[i][object_index] = bad_command[i]
     end
   end
   --print ("@DEBUG - Y valid buff", buff)
end


-- only object related samples in the batch, maybe add some normal samples?
function nql:objLearnMiniBatch(s,a,a_o,bad_command)
  assert(self.transitions:size() > self.minibatch_size)
     function feval()
        self.obj_dw:zero()
        self:setYbuff(a_o, bad_command)
        --print("buffer after setting",self.Y_buff)
        local grad_image = self.Y_buff:ne(0.5) -- maps which gradients we wish to keep
        --print("grad image",grad_image)
        local h_x = self.obj_network:forward(s)
        local J = self.objNetLoss:forward(h_x, self.Y_buff)
        --print("@DEBUG loss calculated "..J, "\npredictions = \n","actual labels= \n",h_x,self.Y_buff) -- just for debugging purpose
        local dJ_dh_x = torch.cmul(self.objNetLoss:backward(h_x, self.Y_buff),grad_image:float():cuda())--zero out none informative gradients
        --print ("after cmul",dJ_dh_x)
        self.obj_network:backward(s, dJ_dh_x) -- computes and updates gradTheta
	      return J, self.obj_dw
     end
     optim.adam(feval, self.obj_w, self.optimState)
end
-- @TODO consider using dedicated aux transition tables for each action to avoid excess resampling
function nql:sample_validation_data()
    print("sampling validation data")
    local s, a, r, s2, term,s_for_obj,a_for_obj,a_o,bad_command = self.transitions:sample(self.valid_size)
    --print(a_o) only none zero objects are sampled for a_o buffer
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()

    self.valid_s_for_obj = s_for_obj:clone()
    self.valid_a_for_obj = a_for_obj:clone()
    self.valid_a_o = a_o:clone()
    self.valid_bad_command = bad_command:clone()
    local bad_parse_samples = bad_command:sum()
    print("total bad parse " .. bad_parse_samples .. " and " .. self.valid_size -  bad_parse_samples.. " succesfull parse")
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self:setYbuff(self.valid_a_o, self.valid_bad_command,true)
    --print(self.valid_Y_buff)
    local h_x = self.obj_network:forward(self.valid_s_for_obj)
    local J = self.objNetLoss:forward(h_x, self.valid_Y_buff)
    local h_y = 1 - h_x:le(0.7) -- calculate prediction, 0.7 value compansates for the strong sample bias for 1 labels
    local sum = 0
    local false_neg = 0
    for i=1,self.valid_size do
      local object_index = self.valid_a_o[i]
      if h_y[i][object_index] == self.valid_bad_command[i] then
        sum = sum + 1
      else
        if self.valid_bad_command[i] == 0 then
          false_neg = false_neg + 1
        end
      end
    end
    --print("predicted lables vs validation sample\n",torch.cat(h_y:float(),self.valid_Y_buff:float()))
    local single_lable_acc = sum/self.valid_size
    print("object net accuracy",single_lable_acc)
    print("obj net is limiting ".. false_neg .. " possible objects")
    print("object net validation loss", J)

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
    return {J, single_lable_acc}
end

--choose an action index based on last s,r,term
function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    --local state = self:preprocess(rawstate):float()
    local state = rawstate:float()
    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)
    --print("@DEBUG: rawstate:\n",rawstate)
    --for i,j in pairs(self.transitions) do
    --  print("@DEBUG:",i,j)
    --end
    --print("@DEBUG: NQL self transitions",self.transitions)

    local currentFullState = self.transitions:get_recent()
    --print("@DEBUG: currentFullState:\n",currentFullState)

    --Store transition s, a, r, s',a_o,bad_command from last step
    if self.lastState and not testing then
			--add(s, a, r, term, a_o,bad_command)
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, self.lastAction_o or 0, self.lastAction_bad)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))
    -- Select action
    local actionIndex = 1
    local a_o
    --print ("@DEBUG: agent state dump: \n",curState)
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)
	  a_o = self.actions[actionIndex].object or 0
		-- given action index we can choose object to work on
    end

    self.transitions:add_recent_action(actionIndex,a_o)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastAction_o = a_o
    self.lastTerminal = terminal
    -- self.lastAction_bad is updated externally after we try the command
    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex,a_o
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local q = self.network:forward(state):float():squeeze()
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]
    self.lastAction_o = self.actions[besta[r]].object or 0
    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    print("@debug create network")
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    local obj_net = self.obj_network
    if self.gpu then
        net:cuda()
        obj_net:cuda()
    else
        net:float()
        obj_net:float()
    end
    return net,obj_net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network, self.obj_network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
    print(get_weight_norms(self.obj_network))
    print(get_grad_norms(self.obj_network))

end
