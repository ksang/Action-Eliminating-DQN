require('ZorkFramework')
opt={}
print ()
local gameEnv = ZorkFramework.GameEnvironment(opt)
print ("init state vec:")
gameEnv:getState()
print("step 1:")
print(gameEnv:step(1,false)) --1 > 2
print("restarting game:\n\n\n\n\n\n\n\n\n\n\n\n\n")
print(gameEnv:newGame())

