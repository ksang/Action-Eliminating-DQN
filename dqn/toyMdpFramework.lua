--- This file defines the class Game environment of a Toy MDP, meant for experimenting with kuz DQN implimentation in lua
local toyMdpFramework = {}

symbols = {}
symbol_mapping = {}
sentence_size = 4
start_index = 1

function parseLine( list_words, start_index)
	-- parse line to update symbols and symbol_mapping
	-- IMP: make sure we're using simple english - ignores punctuation, etc.
	local sindx
	start_index = start_index or 1
	for i=start_index,#list_words do
		word = split(list_words[i], "%a+")[1]
		word = word:lower()
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			symbol_mapping[word] = sindx
		end
	end
end


function text_embedding(input_text)
	local matrix = torch.zeros(sentence_size,#symbols)
	for j, line in pairs(input_text) do
		line = input_text[j]
		local list_words = split(line, "%a+")
    -- check input_text is not longer than sentence_size
    if #list_words <= sentence_size then
    		for i=1,sentence_size do
    			local word = list_words[i]
    			word = word:lower()
    			--ignore words not in vocab
    			if symbol_mapping[word] then
    				matrix[(i-1)*(#symbols) + symbol_mapping[word]] = 1
    			else
    				print(word .. ' not in vocab')
    			end

    		end
    else
      print('number of words in sentence is not' .. sentence_size)
    end
	end
	return matrix
end


-- The GameEnvironment class.
local gameEnv = torch.class('GameEnvironment')

--@ screen: word to vec embedding of the current state in some fixed size buffer (zero padded or truncated)
--@ reward: current score - step number
--@ terminal: dead boolean flag
--@ game_env: object from class GameEnvironment
--@ game_actions: array of actions for agent to affect env
--@ agent: dqn player returned by setup function
--@ opt: arguments passed from terminal when session is launched.

function gameEnv:__init(_opt)
    print("Initializing toy framework")
    self._state.reward = 0
    self._state.terminal = false
    self._state.observation = {}
    self._step_limit = 100
    self._actions= {LEFT = 1,RIGHT = 2}
    self._current_stage = 1
    self._step_penalty = -1
    return self
end
-- be like : game= {{next_stage = {3,2}, descriptor = "bla go right" ,reward= 0,terminal = false },{next_stage = {3,2,3}, descriptor = "bla go left" ,reward= 0,terminal = false }}

--[[ this method ]]
function gameEnv:_updateState(frame, reward, terminal)
    self._state.reward       = reward
    self._state.terminal     = terminal
    self._state.observation  = frame -- in our case frame is the state string descriptor we shold store here the word2vec rep
    return self
end

function gameEnv:getState()
    -- grab the screen again only if the state has been updated in the meantime
    if not self._state.observation then
        self._state.observation = self:_getScreen() -- replace with get current state descriptor vord2vec method
    end
    return self._state.observation, self._state.reward, self._state.terminal -- frame,reward,terminal
end

function gameEnv:newGame()
    self:_updateState({},0,false)
    self._current_stage = 1
    return self:getState()
end
function gameEnv:step(action, training)
    -- accumulate rewards over actrep action repeats
    local cumulated_reward = 0
    local frame, reward, terminal
    for i=1,self._actrep do
        -- Take selected action
        -- maybe change to directly call api_agent's agentStep
        frame, reward, terminal = self:_step(action)

        -- accumulate instantaneous reward
        cumulated_reward = cumulated_reward + reward

        -- game over, no point to repeat current action
        if terminal then break end
    end
    self:_updateState(frame, cumulated_reward, terminal)
    return self:getState()
end


--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    -- return self.api_agent.getStateDims()
    local agent
    if self.onenet then
      agent = self.shooter_agent
    else
      agent = self.api_agent
    end
    return agent.height*agent.width
end


-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
      return self.actions
  end



-- Function plays `action` in the game and return game state.
function gameEnv:_step(action)
    assert(action)
    -- play step with action
    local terminal = 0
    local reward = 0
    local screen = self:_getScreen()

    if self.onenet then
      self.shooter_agent.act(self.d_actions[action][0])
      self.mid_agent.act(self.d_actions[action][1])
      -- have to call both agentStep methods
      self.mid_agent.agentStep()
      terminal = self.shooter_agent.agentStep()
      -- reward is defined as the sum of rewards (for now)
      reward = self.shooter_agent.getReward() + self.mid_agent.getReward()
    else
      self.api_agent.act(action)
      terminal = self.api_agent.agentStep()
      reward = self.api_agent.getReward()
    end
    return screen, reward, terminal
end


function gameEnv:_getScreen()
   ---- This is the NEW WAY ----
   local agent
   if self.onenet then
     agent = self.shooter_agent
   else
     agent = self.api_agent
   end

   local datacount = agent.height * agent.width * 3
   local stringimage = ''

   while datacount > 0 do
     self.img_buffer = self.img_buffer .. agent.img_socket.recv(datacount, 0x40)

     local buflen = string.len(self.img_buffer)
     if buflen == 0 then
       return
     end

     if datacount - buflen >= 0 then
       stringimage = stringimage .. self.img_buffer
       datacount = datacount - buflen
       self.img_buffer = ""
     else
       stringimage = stringimage .. string.sub(self.img_buffer, 1, datacount)
       self.img_buffer = string.sub(self.img_buffer, datacount + 1)
       datacount = 0
     end
   end

   local imstore = torch.ByteStorage():string(stringimage)
   local img_byte_tensor = torch.ByteTensor(imstore, 1, torch.LongStorage{252, 252, 3})
   local img_tensor = img_byte_tensor:type('torch.FloatTensor'):div(255)
   --local img_tensor = image.scale(large_img_tensor, 84, 84)
   --img_tensor:transpose(1,3)
   --print(string.format("Tensor size is %d, %d, %d", img_tensor:size(1), img_tensor:size(2), img_tensor:size(3)))
   --local win = image.display({image=img_tensor, win=win})

   return img_tensor:transpose(1,3) --:transpose(2,3):transpose(1,2)
end

-- Function plays one random action in the game and return game state.
function gameEnv:_randomStep()
    return self:_step(self._actions[torch.random(#self._actions)])
end


return toyMdpFramework
