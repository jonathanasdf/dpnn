--[[
  Fire module as explained in SqueezeNet http://arxiv.org/pdf/1602.07360v1.pdf.
--]]
--FIXME works only for batches.

local FireModule, Parent = torch.class('nn.FireModule', 'nn.Decorator')

function FireModule:__init(nInputPlane, s1x1, e1x1, e3x3, activation, bn)
   self.nInputPlane = nInputPlane
   self.s1x1 = s1x1
   self.e1x1 = e1x1
   self.e3x3 = e3x3
   self.activation = activation or 'ReLU'
   self.bn = bn or true

   if self.s1x1 > (self.e1x1 + self.e3x3) then
      print('Warning: <FireModule> s1x1 is recommended to be smaller'..
            ' then e1x1+e3x3')
   end

   self.module = nn.Sequential()
   self.squeeze = nn.SpatialConvolution(nInputPlane, s1x1, 1, 1)
   self.expand = nn.Concat(2)
   self.expand:add(nn.SpatialConvolution(s1x1, e1x1, 1, 1))
   self.expand:add(nn.SpatialConvolution(s1x1, e3x3, 3, 3, 1, 1, 1, 1))

   local activation_args = self.activation == 'ReLU' and true or nil

   -- Fire Module
   self.module:add(self.squeeze)
   if self.bn then self.module:add(nn.SpatialBatchNormalization(s1x1)) end
   self.module:add(nn[self.activation](activation_args))
   self.module:add(self.expand)
   if self.bn then self.module:add(nn.SpatialBatchNormalization(e1x1 + e3x3)) end
   self.module:add(nn[self.activation](activation_args))

   Parent.__init(self, self.module)
end

--[[
function FireModule:type(type, tensorCache)
   assert(type, 'Module: must provide a type to convert to')
   self.module = nn.utils.recursiveType(self.module, type, tensorCache)
end
--]]

function FireModule:__tostring__()
   return string.format('%s(%d ->  %d -> %d(1x1) + %d(3x3), activation: %s, bn: %s)',
                        torch.type(self), self.nInputPlane, self.s1x1,
                        self.e1x1, self.e3x3, self.activation, tostring(self.bn))
end
