--- This file defines the class Game environment of a Toy MDP, meant for experimenting with kuz DQN implimentation in lua
local toyMdpFramework = {}

local symbols = {}
local symbol_mapping = {}
local sentence_size = 4
local start_index = 1

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


function textEmbedding(input_text)
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
      print('number of words in sentence is larger than' .. sentence_size)
    end
	end
	return matrix
end


-- The GameEnvironment class.
local gameEnv = torch.class('GameEnvironment')
local game = {
    {next_stage = {1,2}, descriptor = "go left" ,reward= 0,terminal = false },        -- 1
    {next_stage = {8,3}, descriptor = "don't go left" ,reward= 0,terminal = false },  -- 2 
    {next_stage = {2,6}, descriptor = "don't go right" ,reward= 0,terminal = false }, -- 3
    {next_stage = {6,5}, descriptor = "go right" ,reward= 0,terminal = false },       -- 4
    {next_stage = {1,6}, descriptor = "go left" ,reward= 0,terminal = false },        -- 5
    {next_stage = {9,5}, descriptor = "don't go left" ,reward= 0,terminal = false },  -- 6
    {next_stage = {3,10}, descriptor = "don't go right" ,reward= 0,terminal = false },-- 7
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
    print("Initializing toy framework")
    self._state.reward = 0
    self._state.terminal = false
    self._state.observation = {}
    self._step_limit = 100
    self._actions= {LEFT = 1,RIGHT = 2}
    self._current_stage = 1
    self._step_penalty = -1
    -- build vocab
	for i=1, #game do
		parseLine(game[i].descriptor)
		end
	end
  return self
end

--[[ this helper method assigns the state arguments ]]
function gameEnv:_updateState(descriptor, reward, terminal)
  self._state.reward       = reward -- @ASK: should this be delta reward or global score ?
  self._state.terminal     = terminal
  self._state.observation  = textEmbedding(descriptor) -- @TODO: in our case frame is the state string descriptor we shold store here the word2vec rep
  return self
end

function gameEnv:getState()
  return self._state.observation, self._state.reward, self._state.terminal -- frame,reward,terminal
end

function gameEnv:newGame()
  self:_updateState(game[1].descriptor ,0,false)
  self._current_stage = 1
  return self:getState()
end

function gameEnv:step(action, training)
  local next_stage, reward, terminal, string
    
  next_stage = game[stage].next_stage[action]
  reward = game[next_stage].reward + this._step_penalty
  terminal = game[next_stage].terminal   
  string = game[next_stage].descriptor 
  self:_updateState(string, reward, terminal)
  return self:getState()
end

--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    -- return self.api_agent.getStateDims()
    return 5*sentence_size -- assume matrix size is
end


-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
      return self._actions
  end

return toyMdpFramework
