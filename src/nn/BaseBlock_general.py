import torch.nn.functional as F
import torch.nn as nn
import torch
from eevbnn.net_bin import Binarize01Act


class BaseBlock_general_kernel_general(nn.Module):


    def __init__(self, args, input_channel, output_channel, kerneltype, channel, t, stride, last=False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """
        super(BaseBlock_general_kernel_general, self).__init__()
        self.stride = stride
        self.args = args
        self.channel = channel
        self.kerneltype = kerneltype
        self.act_bin = Binarize01Act()
        self.act_bin_final = activation_quantize_fn2(a_bit=args.a_bit_final)
        self.input_channel = input_channel
        self.last = last
        # size k * k * g
        self.padding_input = None
        self.t = t
        #print(input_channel,self.g )
        self.output_channel = output_channel
        # for main path:
        c = t * input_channel
        self.group_size =self.channel

        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(output_channel)
        #print(input_channel, c, output_channel, self.group_size)
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size=kerneltype, padding=0, stride=self.stride, bias=True,
                               groups=self.group_size)
        self.conv2 = nn.Conv2d(c, output_channel, kernel_size=1, stride=1, padding=0, bias=True,
                               groups=self.group_size)


    def forward(self, inputs):
        x = inputs
        self.inputs2 = (x+1)/2
        x = F.selu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        if not self.last:
            x = 2 * self.act_bin(x) - 1
        else:
            x = self.act_bin_final(x)
            x = F.relu6(x, inplace=True)
        self.outputblock = x
        return x

class activation_quantize_fn2(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn2, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize2(k=a_bit)

    self.coef = 2**a_bit-1

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.coef * self.uniform_q(torch.clamp(x, 0, 1))
      #print(activation_q)
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q

def uniform_quantize2(k):
  class qfn2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn2().apply