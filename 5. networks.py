import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
    BatchNormalization, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD

# class Network 뇌
# predict
# train_on_batch
# save_model
# load_model
# get_shared_network

# class DNN 뇌의 종류
# get_network_head
# predict
# train_on_batch


# 속성
# shared_network 신경망 상단부(공유부분)
# 함수
# predict 신경망을 통해 투자 행동별 가치 or 확률 계산
# train_on_batch 배치 학습을 위한 데이터 생성
# save_model 신경망 저장하기
# load_model 신경망 불러오기
# get_shared_network 신경망 상단부 생성

class Network:
    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                 shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    def predict(self, sample):  # 샘플생성
        return self.model.predict(sample).flatten()

    def train_on_batch(self, x, y):  # 학습하기
        return  # tf2꺼로바까야함

    def save_model(self, model_path):  # 저장하기
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):  # 불러오기
        if model_path is not None:
            self.model.load_weights(model_path)

    @ classmethod  # 신경망 종류에 따라 공유 신경망 획득하는 함수
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        if net == 'dnn':
            return DNN.get_network_head(Input((input_dim,)))


class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inp = None
        outp = None
        if self.shared_network is None:
            inp = Input((self.input_dim,))
            outp = self.get_network_head(inp).outp
        else:
            inp = self.shared_network.inp
            outp = self.shared_network.outp
        # layer추가로 쌓기
        outp = Dense(
            self.output_dim, activation=self.activation,
            kernel_initializer='random_normal')(outp)
        self.model = Model(input, outp)
        self.model.compile(optimizer=SGD(
            learning_rate=self.lr), loss=self.loss)

    @staticmethod  # 신경망 모델 리턴
    def get_network_head(inp):
        outp = Dense(256, activation='sigmoid',
                     kernel_initializer='random_normal')(inp)
        outp = BatchNormalization()(outp)
        outp = Dropout(0.1)(outp)

        outp = Dense(128, activation='sigmoid',
                     kernel_initializer='random_normal')(outp)
        outp = BatchNormalization()(outp)
        outp = Dropout(0.1)(outp)

        outp = Dense(64, activation='sigmoid',
                     kernel_initializer='random_normal')(outp)
        outp = BatchNormalization()(outp)
        outp = Dropout(0.1)(outp)

        outp = Dense(32, activation='sigmoid',
                     kernel_initializer='random_normal')(outp)
        outp = BatchNormalization()(outp)
        outp = Dropout(0.1)(outp)

        return Model(inp, outp)

    def predict(self, sample):  # 상위클래스 호출
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)

    def train_on_batch(self, x, y):  # 상위클래스 호출
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)


class LSTM(Network):
    pass


class CNN(Network):
    pass
