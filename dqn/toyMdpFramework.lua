--- This file defines the class Game environment of a Toy MDP, meant for experimenting with kuz DQN implimentation in lua
local toyMdpFramework = torch.class('toyMdpFramework')

symbols = {}
symbol_mapping = {}
local sentence_size = 5
local start_index = 1
-- source:
function parseLine( line, start_index)
	-- parse line to update symbols and symbol_mapping
	-- IMP: make sure we're using simple english - ignores punctuation, etc.
	local sindx
  local list_words = string.split(line, " ")
	start_index = start_index or 1
	for i=start_index,#list_words do
		local word = split(list_words[i], "%a+")[1]
		word = word:lower()
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			--print("@DEBUG symbols[sindx]", symbols[sindx])
			symbol_mapping[word] = sindx
			--print("@DEBUG symbol_mapping one", symbol_mapping["go"])
			--print("@DEBUG toy sindx", sindx)
			--print("@DEBUG !!!!! vocab contains",table.unpack(symbols))
			--print("@DEBUG symbol_mapping two",table.unpack(symbol_mapping))
		end
	end
end


function split(s, pattern)
	local parts = {}
	for i in string.gmatch(s, pattern) do
  	table.insert(parts, i)
	end
	return parts
end

function textEmbedding(line)
	--print("@DEBUG text embedding symbol_mapping",symbol_mapping["go"])
	local matrix = torch.zeros(sentence_size,#symbols)
	input_text = string.split(line, " ")
	--print ("@DEBUG: received descriptor for embedding\n",line,table.unpack(input_text))
	for i=1 ,#input_text do
		-- check input_text is not longer than sentence_size, line was truncated
	  if i > sentence_size then
			print('number of words in sentence is larger than' .. sentence_size)
			break
		end

		local word = input_text[i]
		local normlized_word = split(word, "%a+")[1]
		normlized_word = normlized_word:lower()
		--print("@DEBUG",symbol_mapping[normlized_word])
		--ignore words not in vocab
  	if symbol_mapping[normlized_word] then
			matrix[i][symbol_mapping[normlized_word]] = 1
		else
			print(normlized_word .. ' not in vocab')
		end
	end
	--print ("@DEBUG: generated state descriptor embedding\n",matrix)
	return matrix
end


-- The GameEnvironment class.
local gameEnv = torch.class('toyMdpFramework.GameEnvironment')
local game = {
    {next_stage = {1,2}, descriptor = "go left" ,reward= 0,terminal = false },        -- 1
    {next_stage = {8,3}, descriptor = "dont go left" ,reward= 0,terminal = false },  -- 2
    {next_stage = {2,6}, descriptor = "dont go right" ,reward= 0,terminal = false }, -- 3
    {next_stage = {6,5}, descriptor = "go right" ,reward= 0,terminal = false },       -- 4
    {next_stage = {1,6}, descriptor = "go left" ,reward= 0,terminal = false },        -- 5
    {next_stage = {9,5}, descriptor = "dont go left" ,reward= 0,terminal = false },  -- 6
    {next_stage = {3,10}, descriptor = "dont go right" ,reward= 0,terminal = false },-- 7
    {next_stage = {7,3}, descriptor = "go right" ,reward= 0,terminal = false },       -- 8
    {next_stage = {4,10}, descriptor = "go left" ,reward= 0,terminal = false },       -- 9
    {next_stage = {10,10}, descriptor = "win" ,reward= 10,terminal = true }           --10
}
--@ screen: word to vec embedding of the current state in some fixed size buffer (zero padded or truncated)
--@ reward: current score - step number
--@ terminal: dead boolean flag
--@ game_env: object from class GameEnvironment
--@ game_actions: array of actions for agent to affect env
--@ agent: dqn player returned by setup function
--@ opt: arguments passed from terminal when session is launched.

function gameEnv:__init(_opt)
    print("@DEBUG Initializing toy framework")
		-- build vocab
		for i=1, #game do
			parseLine(game[i].descriptor)
		end
		self._state={}
		self._state.reward = 0
    self._state.terminal = false
		self._state.observation = textEmbedding("go left")
		--print("@DEBUG",self._state.observation )
		self._step_limit = 100
		local LEFT = 1
		local RIGHT = 2
		self._actions= {LEFT,RIGHT}
		self._current_stage = 1
    self._step_penalty = -1
		--print("@DEBUG vocab contains ",table.unpack(symbols))
		--print("@DEBUG symbol_mapping ",table.unpack(symbol_mapping)) -cannot print, but can use symbol_mapping
		--print("@DEBUG Initializing toy framework -- DONE!")

  return self
end

--[[ this helper method assigns the state arguments ]]
function gameEnv:_updateState(descriptor, reward, terminal)
	-- print("@DEBUG _updateState")
	self._state.reward       = reward -- @ASK: should this be delta reward or global score ?
  self._state.terminal     = terminal
	self._state.observation  = textEmbedding(descriptor) -- @TODO: in our case frame is the state string descriptor we shold store here the word2vec rep
  return self
end

function gameEnv:getState()
  print("@DEBUG toy getState")
	print("@DEBUG toy getState returned state\n",self._state.observation)
	return self._state.observation, self._state.reward, self._state.terminal -- frame,reward,terminal
end

function gameEnv:newGame()
	print("@DEBUG toy newGame")
	self:_updateState(game[1].descriptor ,0,false)
  self._current_stage = 1
  return self:getState()
end

function gameEnv:step(action, training)
	print("@DEBUG toy step")
	local next_stage, reward, terminal, string
	--print("@DEBUG: agent selected action w index",action)
  next_stage = game[self._current_stage].next_stage[action]
	--print ("@DEBUG: current stage is " ,self._current_stage,"next stage is",next_stage)
  self._current_stage = next_stage
	reward = game[next_stage].reward + self._step_penalty
  terminal = game[next_stage].terminal
  string = game[next_stage].descriptor
  print("@DEBUG toy step results:\n","action selected: ",action," step reward: ",reward, "next_stage: ",next_stage)
	self:_updateState(string, reward, terminal)
  return self:getState()
end

--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    -- return self.api_agent.getStateDims()
		local dim_size = torch.Tensor(2)

		dim_size[1] = 5
		dim_size[2] = 5

		return dim_size -- assume matrix size is 5*5

end

-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
      return self._actions
end

return toyMdpFramework
