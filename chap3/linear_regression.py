import torch
import matplotlib.pyplot as plt

torch.manual_seed(777)

lr = 0.01
iterations = 1000

a = torch.tensor(1)
# generate.data
x = torch.rand(200,1)*10
y = 3*x + (5 + torch.randn(200,1))

w = torch.randn(1,requires_grad = True)
b = torch.zeros(1,requires_grad = True)
best_loss = float('inf')

plt.ion()

fir, ax = plt.subplots()
plt.rcParams['lines.markersize'] = 3
for iteration in range(iterations):
    wx = torch.mul(w,x)
    y_pred = torch.add(wx,b)
    loss = (0.5*(y-y_pred)**2).mean()
    loss.backward()
    b.data -= lr*b.grad
    w.data -= lr*w.grad
    w.grad.data.zero_()
    b.grad.data.zero_()

    current_loss = loss.item()

    plt.title('iteration: {}\nw: {} b: {}'\
              .format(iteration,w.data.numpy(),b.data.numpy()),
              fontsize = 'xx-large'
              )
    plt.xlim(1.5,40)
    plt.ylim(8,40)
    plt.plot(x.data.numpy(),y_pred.data.numpy(),'r-',lw=5)
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.text(2,20,'Loss=%.4f' % loss.data.numpy(),
             fontdict={'size':20,'color':'red'})
    plt.pause(0.1)
    ax.cla()