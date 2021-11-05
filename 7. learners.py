import os
import logging
import abc
import collections
import time
import numpy as np
from tqdm import tqdm

from settings import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

# ReinforcementLearner 학습기
# init_value_network 가치결정망
# init_policy_network 정책결정망
# reset 리셋
# build_sample 샘플
# get_batch
# update_networks 배치 학습데이터 생성
# fit 신경망 갱신
# visualize 시각화
# run 실행
# save_models 모델 저장
# predict

# DQNLearner DQN
# get_batch


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta  # 추상클래스. 객체생성불가

    def __init__(self, rl_method='rl', stock_code=None, chart_data=None, training_data=None,
                 min_trading_unit=1, max_trading_unit=2, delayed_reward_threadshold=.05,
                 net='dnn', num_steps=1, lr=0.001, discount_factor=0.9,
                 num_epoches=100, balance=1000000, start_epsilon=1,
                 value_network=None, policy_network=None, output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, balance,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리 여기 저장한 보상으로 신경망 학습 진행. 학습 데이터 샘플, 행동, 보상, 예측 가치, 예측 확률, 포폴 가치, 보유 주식 수, 탐험 위치, 학습 위치
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 손실, 수익발생 횟수, 탐험 횟수, 학습횟수 등 기록해서 output_path에 저장
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

        # 가치 신경망 생성 함수
        def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
            if self.net == 'dnn':
               self.value_network = DNN(
                    input_dim=self.num_features,
                    output_dim=self.agent.NUM_ACTIONS,
                    lr=self.lr,
                    shared_network=shared_network,
                    activation=activation,
                    loss=loss)

        # 정책 신경망 생성 함수
        def init_policy_network(self, shared_network=None, activation='sigmoid', loss='binary_crossentropy'):
            if self.net == 'dnn':
                self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                shared_network=shared_network,
                activation=activation,
                loss=loss)
                )

    def reset(self):  # 1에폭 후 전체 초기화
        self.sample=None
        self.training_data_idx=-1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample=[]
        self.memory_action=[]
        self.memory_reward=[]
        self.memory_value=[]
        self.memory_policy=[]
        self.memory_pv=[]
        self.memory_num_stocks=[]
        self.memory_exp_idx=[]
        self.memory_learning_idx=[]
        # 에포크 관련 정보 초기화
        self.loss=0.
        self.itr_cnt=0
        self.exploration_cnt=0
        self.batch_size=0
        self.learning_cnt=0

    def build_sample(self):  # 샘플 만들기
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample=self.training_data.iloc[self.training_data_idx].tolist(
            )
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @ abc.abstractmethod
    def get_batch(self): # 강제성
        pass

    def update_networks(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy=self.get_batch()
        if len(x) > 0:
            loss=0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    def fit(self):
        # 배치 학습 데이터 생성 및 신경망 갱신
        _loss=self.update_networks()
        if _loss is not None:
            self.loss += abs(_loss)
            self.learning_cnt += 1
            self.memory_learning_idx.append(self.training_data_idx)

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action=[Agent.ACTION_HOLD] * \
            (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks=[0] * \
            (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value=[np.array(  # 확인하기
                [np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy=[np.array(
                [np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_policy
        self.memory_pv=[self.agent.initial_balance] * \
            (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str = epoch_str, num_epoches = num_epoches,
            epsilon = epsilon, action_list = Agent.ACTIONS,
            actions = self.memory_action,
            num_stocks = self.memory_num_stocks,
            outvals_value = self.memory_value,
            outvals_policy = self.memory_policy,
            exps = self.memory_exp_idx,  # 확이하기
            learning_idxes = self.memory_learning_idx,
            initial_balance = self.agent.initial_balance,
            pvs = self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir,
            'epoch_summary_{}.png'.format(epoch_str))
        )

    def run(self, learning = True):
        info=(
            "[{code}] RL:{rl} Net:{net} LR:{lr} "
            "DF:{discount_factor} TU:[{min_trading_unit},{max_trading_unit}]"
        ).format(
            code = self.stock_code, rl = self.rl_method, net = self.net,
            lr = self.lr, discount_factor = self.discount_factor,
            min_trading_unit = self.agent.min_trading_unit,
            max_trading_unit = self.agent.max_trading_unit,
        )
        with self.lock:
            logging.info(info) # 계획이 잘 진행되고 있음을 보여줌

        # 시작 시간
        time_start=time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir=os.path.join(
            self.output_path, 'epoch_summary_{}'.format(self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 학습에 대한 정보 초기화
        max_portfolio_value=0
        epoch_win_cnt=0

        # 학습 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch=time.time()

            # step 샘플을 만들기 위한 큐
            q_sample=collections.deque(maxlen = self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon=10 / \
                    (epoch + 10) if epoch < self.num_epoches - 1 else 0
                self.agent.reset_exploration()
            else:
                epsilon=self.start_epsilon
                self.agent.reset_exploration(alpha = 0)

            for i in tqdm(range(len(self.training_data))):
                # 샘플 생성
                next_sample=self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장. q샘플 계속 돌리다가 다 차면 내려감
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측.확인
                pred_value=None
                pred_policy=None
                if self.value_network is not None:
                    pred_value=self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy=self.policy_network.predict(list(q_sample))

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration=self.agent.decide_action(
                    pred_value, pred_policy, epsilon)

                # 즉시 보상 획득
                reward=self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            # 에포크 종료 후 학습
            if learning:
                self.fit()

            # 에포크 관련 정보 로그 기록
            num_epoches_digit=len(str(self.num_epoches))
            epoch_str=str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch=time.time()
            elapsed_time_epoch=time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                         "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                         "#Stocks:{} PV:{:,.0f} "
                         "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                             self.stock_code, epoch_str, self.num_epoches, epsilon,
                             self.exploration_cnt, self.itr_cnt,
                             self.agent.num_buy, self.agent.num_sell,
                             self.agent.num_hold, self.agent.num_stocks,
                             self.agent.portfolio_value, self.learning_cnt,
                             self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            if self.num_epoches == 1 or (epoch + 1) % 10 == 0:
                self.visualize(epoch_str, self.num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value=max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1 # 수식 확인

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                         "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                             code=self.stock_code, elapsed_time=elapsed_time,
                             max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self, balance=10000000):
        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)

        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장. 무엇?
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))

            # 신경망에 의한 행동 결정
            action, confidence, _ = self.agent.decide_action(
                pred_value, pred_policy, 0)

            result.append((action, confidence))

        return result

class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))

        value_max_next = 0
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward

            y_value[i] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            value_max_next = value.max()

        return x, y_value, None

class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward

            if r > 0:
                y_policy[i, action] = 1
            else:
                y_policy[i, action] = 0

        return x, None, y_policy

class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None,
                 value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps,
                input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample),
                     self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, action] = r + self.discount_factor * value_max_next
            # advantage = y_value tf2
            y_policy[i, action] = 1 if r > 0 else 0
            value_max_next = value.max()
        return x, y_value, y_policy

