--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'
require 'nnutils'
require "initenv"


return function(args)
  assert(args.n_objects)
  local in_row_s = 65
  local in_col = 300
  local in_hist = 4
  local input_dims_s = {in_hist,in_row_s,in_col}
  local input_dims_a = {1,in_row_a,in_col}
  --local input_dims = {in_hist,in_row,in_col}
  --local n_actions = 2
  local region_hight = {1,2,3} -- hight only of filter, width will be 'in_col'
  local n_filters = 2 -- number of filters per region size
  local tot_filters_s = table.getn(region_hight)*n_filters
  local tot_filters_a = 1*n_filters
  local output_size = args.n_objects
  ----------------------------
  -- NETWORK FOR STATES ONLY
  ----------------------------

  local conv_s = nn.DepthConcat(4)
  conv_s:add(nn.SpatialConvolution(4,n_filters,in_col,region_hight[1]))
  conv_s:add(nn.SpatialConvolution(4,n_filters,in_col,region_hight[2]))
  conv_s:add(nn.SpatialConvolution(4,n_filters,in_col,region_hight[3]))

  local net_s = nn.Sequential()
  --net_s:add(nn.Reshape(in_row_s*in_col*in_hist))
  --net_s:add(nn.Linear(in_row_s*in_col*in_hist,7))
  net_s:add(nn.Reshape(unpack(input_dims_s)))
  net_s:add(conv_s)
  -- @DEBUG: output size is
  -- ((32 or 1)*2)Xn_filtersX(((in_row-min(region_hight)+2*0)/1)+1)X1
  net_s:add(nn.ReLU())
  net_s:add(nn.SpatialMaxPooling(1,((in_row_s-math.min(unpack(region_hight))+2*0)/1)+1))
  net_s:add(nn.Reshape(tot_filters_s))
  net_s:add(nn.Linear(tot_filters_s,output_size))
  net_s:add(nn.Sigmoid())

  --[[local net_s = nn.Sequential()
  net_s:add(nn.Reshape(in_row_s*in_col*in_hist))
  net_s:add(nn.Linear(in_row_s*in_col*in_hist,emb_length))]]
  print(net_s)
  return net_s
end
