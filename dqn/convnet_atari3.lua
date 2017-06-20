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
  network:add(nn.Reshape(4*3*300)) --@DEBUG_DIM(hist*state (sentence) size*word representation)
  network:add(nn.Linear(4*3*300,10)) --@DEBUG_DIM(hist*state (sentence) size*word representation)
  network:add(nn.Linear(10,2))
    return network
end
