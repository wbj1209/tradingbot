# class Environment
# reset 관측값, 위치 초기화
# observe 관측값 이동
# get price 관측값에서 종가 획득
# set chart data 차트데이터 가져오기

class Environment:
    # 환경 상태
    PRICE_IDX = 4

    def __init__(self, chart_data=None):
        self.chart_data = chart_data  # 차트 데이터를 입력으로 받아 객체 내 변수(chart_data)에 저장
        self.observation = None  # 현재 위치의 관측값
        self.idx = -1  # 현재 위치

    def reset(self):  # 초기화
        self.observation = None
        self.idx = -1

    def observe(self):  # 관측
        if len(self.chart_data) > self.idx + 1:  # 차트 데이터의 전체 길이보다 다음 위치가 작다면 가져올 데이터가 있다
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation  # chart_data 1줄
        return None

    def get_price(self):  # 관측값
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):  # 차트
        self.chart_data = chart_data
