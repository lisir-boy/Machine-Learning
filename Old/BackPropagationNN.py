import math    #math 更多https://docs.python.org/zh-cn/3/library/index.html
import random   #导入生成伪随机数器
import numpy as np  #导入矩阵库
np.seterr(all = 'ignore')

###NN神经网络关键是非线性的激活函数 因为如果激活函数也是线性的 则应用神经网络没有优势，完全可以用线性拟合进行拟合！！！！！
# sigmoid transfer function
# IMPORTANT: when using the logit (sigmoid) transfer function for the output layer make sure y values are scaled from 0 to 1
# if you use the tanh for the output then you should scale between -1 and 1
# we will use sigmoid for the output layer and tanh for the hidden layer
#该例子中只有一层隐藏层
def sigmoid(x):  #输出激活函数
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):   #输出激活函数导数
    return y * (1.0 - y)

# using tanh over logistic sigmoid is recommended   
def tanh(x):  #隐藏层激活函数
    return math.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):  #导数
    return 1 - y*y

class MLP_NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network （多层感知器）, adapted and from the book 'Programming Collective Intelligence' (http://shop.oreilly.com/product/9780596529321.do)
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm（算法） has been generalized to be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition识别 dataset provided by sklearn
    Fully pypy compatible.完全兼容PYPY
    """
    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay #衰退率
        
        # initialize arrays
        self.input = input + 1 # add 1 for bias node 
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations 建立元素为1的数组
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1/2) #正态分布的标准差
        output_range = 1.0 / self.hidden ** (1/2)  #同上
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))  #生成正态分布数组，数组大小为（self.input行, self.hidden列）
        self.wo = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output)) #同上
        
        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration迭代
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))  #生成self.input行，self.hidden列的元素为0的数组
        self.co = np.zeros((self.hidden, self.output))  #同上

    def feedForward(self, inputs):  #向前传递
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data  #与--init--的参数不同 是复数 多"S"
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')  #输入错误你个傻蛋

        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias 除去偏差
            self.ai[i] = inputs[i] #将输入的数据赋给ai

        # hidden activations  #只有一层中间层
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)  #ah[j]中存放的是中间层的值 tanh()是中间层激活函数

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)  #输出层的输出结果 sigmoid()是输出层的激活函数

        return self.ao[:]

    def backPropagate(self, targets):  #向后传播
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (Wrong number of inputs you silly goose) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative

        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])  #error是误差值
            output_deltas[k] = dsigmoid(self.ao[k]) * error  #因为loss=1/2*[(y真-y预测)+……]^2（由156行代码看出）  所以对‘输出的sum’求导是该式 
                                                             #而‘输出sum’对wo[j][k]求导是ah[j]

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]  #求loss对‘hidden层的ah[j]’的导为此表达式   
            hidden_deltas[j] = dtanh(self.ah[j]) * error  #求loss对‘hidden层的sum’的导为此表达式    而‘hidden层的sum’对wi[i][j]求导是ai[i]

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]  #loss对wo[j][k]的导
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum  #此处self.momentumi写错了没有'i',已改 
                                                                                               #加co这一项是为了更快收敛？看最优化书
                self.co[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]  #loss对wi[i][j]的导
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum  
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2  #计算LOSS值
        return error  #向后传递函数最终返回得到的是损失值 即LOSS值
'''
    def test(self, patterns):  #测试程序用
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))
'''
    def train(self, patterns):
        # N: learning rate
        for i in range(self.iterations):  #迭代次数
            error = 0.0
            random.shuffle(patterns)  #对patterns进行随机排序 排完序 名称还是patterns
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            if i % 10 == 0:
                print('error %-.5f' % error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
                
    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions

def demo():
    """
    run NN demo on the digit recognition dataset from sklearn
    """
    #输入数据的处理
    def load_data():
        data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')

        # first ten values are the one hot encoded y (target) values
        y = data[:,0:10]
        #y[y == 0] = -1 # if you are using a tanh transfer function make the 0 into -1
        #y[y == 1] = .90 # try values that won't saturate tanh
        
        data = data[:,10:] # x data
        #data = data - data.mean(axis = 1)
        data -= data.min() # scale the data so values are between 0 and 1  #data是多维数组变量 numpy.ndarray类型
                                                                           #如果输出为该程序的内存位置 意味着没有写该函数的参数即括号
                                                                           #元组 列表 与ndarray不一样
                                                                           
        data /= data.max() # scale
        
        out = []  #类型为list
        print(data.shape）  #输出矩阵data的大小  不加括号为python2的规则

        # populate the tuple list with the data  #将数据填充到元组列表中
        for i in range(data.shape[0]):
            fart = list((data[i,:].tolist(), y[i].tolist())) # don't mind this variable name    fart译为放屁
            out.append(fart)  #在Out数组末尾添加元素 fart  最终类似于[[[0.29629629654320994, 0.6666666665555556, 1.0], [5.0, 86.0]], [[ 0.6666666665555556, 0.8333333332777779, 1.0], [12.0, 233.0]]]
                              #而在前面定义的train（）中random.shuffle（）只是将列表最外面一层进行随机排序 也就是每组内的对应关系不变
        return out

    X = load_data()

    print X[9] # make sure the data looks right

    NN = MLP_NeuralNetwork(64, 100, 10, iterations = 50, learning_rate = 0.5, momentum = 0.5, rate_decay = 0.01)

    NN.train(X)

    NN.test(X)

if __name__ == '__main__':
    demo()
