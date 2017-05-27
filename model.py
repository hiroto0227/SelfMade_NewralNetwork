import numpy as np

class Data():
#Data.X_train
#Data.y_train
#Data.X_test
#Data.y_test
#Data.train_image_count
#Dara.test_image_count

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_image_count = None
        self.test_image_count = None

        self.loadTrainData()
        self.loadTrainLabel()
        self.loadTestData()
        self.loadTest

    def loadTrainData(self):
        with open('./data/train-images-idx3-ubyte', 'rb') as inputfile:
            content = inputfile.read()
        self.train_image_count = int.from_bytes(content[4:8], 'big') # 手書き画像枚数
        image_row = int.from_bytes(content[8:12], 'big') # 手書き画像行数
        image_col = int.from_bytes(content[12:16], 'big') # 手書き画像列数
        images = np.frombuffer(content, np.uint8, -1, 16)# 手書き画像データの読み込み
        self.X_train = images.reshape(self.train_image_count, image_row, image_col, 1) # 一つの画像を一列にまとめる

    def loadTrainLabel(self):
        with open('./data/train-labels-idx1-ubyte', 'rb') as inputfile:
            content = inputfile.read()
        labels = np.frombuffer(content, np.uint8, -1, 8) #ラベルの読み込み(5,0,4,...)
        one_hot_labels = np.zeros([self.train_image_count, 10])  #one_hot_vectorに変換
        for label in labels:
            one_hot_labels[np.arange(self.train_image_count), label] = 1
        self.y_train = one_hot_labels


class NeuralNetwork():

    def __init__(self):
        self.hidden_dim = 400
        self.output_dim = 10

    def fit(self, X_train, y_train):
        num_example = X_train.shape[0]
        input_dim = X_train.shape[1] * X_train.shape[2]
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim

        w1 = np.zeros((hidden_dim, input_dim))
        b1 = np.zeros((hidden_dim, 1))
        w2 = np.zeros((output_dim, hidden_dim))
        b2 = np.zeros((output_dim, 1))

        for i in range(num_example):
            print("{}/ 100 % ".format(i / num_example * 100))
            #initialize parameter
            x = X_train[i].reshape((input_dim, 1))
            Y = y_train[i].reshape([output_dim, 1])

            #Forward Propagation
            print("Forward Propagation")
            h = np.tanh(w1.dot(x) + b1) #input→hidden
            o = w2.dot(h) + b2 #hidden→output
            exp_o = np.exp(o) 
            y = exp_o / np.sum(exp_o) #softmax 

            #Back propagation
            print("Back Propagation")
            L = -1 * np.sum(Y * np.log(y)) #Loss function for cross entropy and softmax function
            delta_out = y - Y #Error for output→hidden
            delta_hidden = (1-np.power(h, 2)) * w2.T.dot(delta_out) #Error for hidden→input

            #update rate
            dw2 = delta_out.dot(h.T)
            db2 = delta_out
            dw1 = delta_hidden.dot(x.T)
            db1 = delta_hidden

            #update parameter
            epsilon = 0.1 + 1 / (i + 1) #はじめは大きく、どんどん小さく。
            w2 += epsilon * dw2
            b2 += epsilon * db2
            w1 += epsilon * dw1
            b1 += epsilon * db1
        
        print(w1)
    
    def predict(self, X_test, y_test):
        


if __name__ == '__main__':
    print("Data Loading")
    data = Data()
    model = NeuralNetwork()
    print("Training Start!")
    model.fit(data.X_train, data.y_train)