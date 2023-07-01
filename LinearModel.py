import numpy as np
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader

lr=0.01

class MSELoss:
    def __init__(self):
        self.derivative=0
        pass

    def backward(self):
        return self.derivative
    
    def __call__(self, predict, target):
        elements_num = 1
        for shape in predict.shape:
            elements_num *= shape
        mse_loss = np.sum((predict - target) ** 2) / elements_num

        self.derivative = predict - target

        return mse_loss
    
class Sigmoid:
    def __init__(self):
        self.z = None
    def backward(self):
        return self.z * (1 - self.z)
    def __call__(self, x):
        self.z = 1 / (1 + np.exp(-x))
        return self.z

class ReLU:
    def __init__(self):
        self.derivative=0
        pass
    def backward(self):
        return self.derivative
    def __call__(self, x):
        if x>0:
            self.derivative=1
        else:
            self.derivative=0
        return max(x, 0)

class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_matrix = np.random.random((input_dim, output_dim))
        self.bias = np.random.random(output_dim)
        self.x=0
        self.activation = Sigmoid()
    
    def __call__(self, input_matrix):
        assert input_matrix.shape[-1] == self.weight_matrix.shape[0], f"input dimension and output dimension should be the same. \n now : input_dim = {input_matrix.shape[-1]}, weight_dim = {self.weight_matrix.shape[0]}"
        self.x = input_matrix

        result = np.dot(input_matrix, self.weight_matrix) + self.bias
        print(f"self.x : {self.x[0][-1]}, result : {result[0][-1]}, act : {self.activation(result)[0][-1]}")
        return self.activation(result)

    def backprop(self, dL_dz): # 16,31
        # 가중치 : w, bias : b, 입력값 : x, 출력값 : z, 로스 미분값 : dL
        print(f"dL_dz : {dL_dz[0][-1]}")
        dz_dy = self.activation.backward() # 16, 31
        print(f"dz_dy : {dz_dy[0][-1]}")
        dy_dw = self.x # 16,32
        dw = np.dot(dy_dw.T, dz_dy*dL_dz)
        db = 1*np.sum(dz_dy*dL_dz, axis=0)
        self.weight_matrix -= lr*dw
        print(self.weight_matrix[0][-1])
        self.bias -= lr*db

def main():
    criterion = MSELoss()

    test_input=np.random.randint(1, 100, size=(16,32))
    target = np.ones((16,31))+5

    Layer = Linear(32, 31)
    
    for i in range(500):
        predict = Layer(test_input)
        mse_loss = criterion(predict, target)
        Layer.backprop(criterion.backward())

        print("mse_loss : ", mse_loss)
        # if i%100==99:

    # batch_size = 16

    # train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    # test_dataset = MNIST(root='./data', train=False, transform=ToTensor())

    # # DataLoader는 우선 나중에 구현하기
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    main()