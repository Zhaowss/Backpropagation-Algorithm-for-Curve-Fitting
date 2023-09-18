import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 输入数据（时间）
x = np.array([1, 2, 3, 4, 5, 6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,23, 24,25,26]).astype(float)
mydata=pd.read_excel("学习曲线.xlsx")

X1=mydata.iloc[:,1]
X2=mydata.iloc[:,2]
X3=mydata.iloc[:,3]
X4=mydata.iloc[:,4]
print(X1)
x_fit = np.linspace(0, 25, 26)
import numpy as np
import matplotlib.pyplot as plt





# 输入数据，这里假设你有26个数据点
data = X4

# 目标值或预期值
mu_0 = np.mean(data)

# 控制参数（可以根据需要进行调整以控制敏感度）
k = 0.5

# 计算CUSUM值
cusum = np.zeros(len(data))
for i in range(1, len(data)):
    cusum[i] = max(0, cusum[i-1] + (data[i] - mu_0 - k))


y = np.array(cusum)
# 输入数据
x = torch.tensor(x_fit, dtype=torch.float32)
# 实验结果（目标值）
y = torch.tensor(y, dtype=torch.float32)

print(x.shape)
print(y.shape)
print(y)

class PolynomialModel(torch.nn.Module):
    def __init__(self, degree):
        super(PolynomialModel, self).__init__()
        self.degree = degree
        self.weights = torch.nn.Parameter(torch.randn(degree + 1, requires_grad=True))

    def forward(self, x):
        powers = torch.arange(self.degree + 1, dtype=torch.float32)
        x_powers = x.unsqueeze(1) ** powers  # 计算 x 的幂次项
        return torch.sum(self.weights * x_powers, dim=1)

# 损失函数
def loss_function(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

degree =4
model = PolynomialModel(degree)
optimizer = optim.Adam(model.parameters(), lr=3.2)
num_epochs = 100000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

coefficients = model.weights.detach().numpy()

plt.scatter(x.numpy(), y.numpy(), label='Original Data')
x_fit = torch.linspace(min(x), max(x), 26)
y_fit = model(x_fit)
plt.plot(x_fit.numpy(), y_fit.detach().numpy(), 'r', label='Fitted Curve')
plt.xlabel('case number')  # 替换为您希望的横轴标签
plt.ylabel('CUSUM of blood loss(ml)') 
# 显示多项式方程
coefficients = model.weights.detach().numpy()
equation = 'y='
for i, coeff in enumerate(coefficients):
    equation += f'{coeff:.2f}x^{i} + '
equation = equation[:-3]  # 去掉最后的 '+ '
plt.text(5, min(y.numpy())+70, equation, fontsize=12, verticalalignment='bottom')

xx=''
# 计算并添加R²值
r2 = r2_score(y.numpy(), y_fit.detach().numpy())
xx += f'\nR² = {r2:.2f}'

plt.text(8, min(y.numpy())+60, xx, fontsize=12, verticalalignment='bottom')


plt.legend(fontsize=12)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)


plt.show()













