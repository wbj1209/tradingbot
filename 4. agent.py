import numpy as np

from settings
# class agent 손
# reset 에이전트 상태 초기화
# reset exploration 탐험 기준을 정함
# set balance 자본금 설정
# get states 에이전트 상태 획득. 주식비율 무엇?
# decide action 행동 결정. 신경망할지 랜덤할지 이거 무엇?
# validate action 행동 유효성 검사. 살수 있는 잔액이 있는지, 팔수있는 주식이 있는지
# decide trading unit 트레이딩 볼륨 결정. add trading 무엇?
# act 매수, 매도, 홀딩 . 손절은 어케?


class Agent:
    # 에이전트 상태
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율
    TRADING_CHARGE = 0.015  # 거래 수수료
    TRADING_TAX = 0.25  # 거래세

    # 에이전트 행동
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, environment, balance, min_trading_unit=1, max_trading_unit=2):
        self.environment = environment

        self.initial_balance = balance  # 초기 자본금
        self.balance = balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # PV: balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준

        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위

    def reset(self):  # 초기화
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):  # 탐험 기준 갱신
        self.exploration_base = 0.5 + np.random.rand() / 2  # 매수 탐험 선호를 위해 50% 확률 부여

    def set_balance(self, balance):  # 초기 자본금 설정
        self.initial_balance = balance

    def get_states(self):  # 에이전트의 상태를 획득
        self.ratio_hold = self.num_stocks / \
            int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value)

        return self.ratio_hold, self.ratio_portfolio_value

    # pred[action], np.random.rand() < epsilon, self.NUM_ACTIONS - 1) + 1
    def decide_action(self, pred_value, pred_policy, epsilon):  # 행동 결정
        confidence = 0.  # pred_policy값은 그대로, value_policy값은 signoid이용해서 쓴다

        pred = pred_policy  # pred_policy = [확률값, 확률값]
        if pred is None:
            pred = pred_value  # pred_value = [값, 값]

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():  # 정책을판단할수없을때(매수매도 우열 못가릴때 탐험해라)
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY  # 0
            else:
                action = np.random.randint(
                    self.NUM_ACTIONS - 1) + 1  # 1 or 2 관망이나 매도
        else:
            exploration = False
            action = np.argmax(pred)  # 가치 or 정책의 argmax로 행동하자

        confidence = .5  # 아래 둘다 없을때 이거 쓰려고
        if pred_policy is not None:  # 정책네트웍 있으면 위에서 0 1 2 정해진걸로
            confidence = pred[action]  # confidence구할수있어
        elif pred_value is not None:
            confidence = settings.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):  # 행동의 유효성 판단
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                return False
        return True

    # confidence
    def decide_trading_unit(self, confidence):  # 행동의 볼륨 판단
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)),
                                self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):  # 행동
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            trading_unit = self.decide_trading_unit(confidence)
            balance = (self.balance - curr_price *
                       (1 + self.TRADING_CHARGE) * trading_unit)
            if balance < 0:
                trading_unit = max(min(int(self.balance / (curr_price * (
                    1 + self.TRADING_CHARGE))), self.max_trading_unit), self.min_trading_unit)
            invest_amount = curr_price * \
                (1 + self.TRADING_CHARGE) * trading_unit

            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            trading_unit = self.decide_trading_unit(confidence)
            trading_unit = min(trading_unit, self.num_stocks)
            invest_amount = curr_price * \
                (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit

            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) / self.initial_balance)

        self.immediate_reward = self.profitloss

        delayed_reward = 0

        self.base_profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        if self.base_profitloss > self.delayed_reward_threshold or \
           self.base_profitloss < -self.delayed_reward_threshold:
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
