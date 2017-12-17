import torch
import numpy as np
from torch.autograd import Variable

batch_size = 10
input_dimension  = 100
hidden_dimension = 500
output_size = 10

x = Variable(torch.FloatTensor(batch_size,input_dimension), requires_grad=False)
y = Variable(torch.FloatTensor(batch_size, output_size), requires_grad=False)

w1 = Variable(torch.FloatTensor(input_dimension, hidden_dimension), requires_grad=True)
w2 = Variable(torch.FloatTensor(hidden_dimension, output_size), requires_grad=True)

learning_rate = 0.0001
epochs = 500

for epoch in range(epochs):

    y_out = x.mm(w1).clamp(min=0).mm(w2)
    print "y_out.data",y_out.data[:10]
    print "its size",y_out.size
    loss = (y - y_out).pow(2).sum()
    print "loss",loss.data

    loss.backward()
    w1.grad.data.zero_()
    w2.grad.data.zero_()

    try:
        print "x-grad",x.grad
        print "y-grad",y.grad
        print "w1-grad",w1.grad
        print "w2_grad",w2.grad
    except:
        print "w1-grad",w1.grad
        print "w2_grad",w2.grad

    w1 = w1 - w1.grad*learning_rate
    w2 = w2 - w2.grad*learning_rate

    print "{} loss at {}th epoch".format(loss.data,epoch)
