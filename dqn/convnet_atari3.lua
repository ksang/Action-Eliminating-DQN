--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'convnet'

--[[return function(args)
    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {512}
    args.nl             = nn.Rectifier

    return create_network(args)
end]]

return function(args)
  --FIXME history was set in run_gpu to 4 via opt
  local network = nn.Sequential()
  network:add(nn.Reshape(4*5*5))
  network:add(nn.Linear(100,10))
  network:add(nn.Linear(10,2))
  print('hi this is convnet atari 3 !!! \n')
  print(network:size())
  print('hi this is convnet atari 3  second time!!! \n')
    return network
end
