import os
import sys
import logging  # 디버깅
import argparse  # argparse.ArgumentParser()은 add_argument()로 위치 인자, 키워드 인자 추가
import json  # loads() 문자열 -> 딕셔너리, dumps() 딕셔너리 -> 문자열

import settings
import data_manager

# if __name__ == '__main__':
# 출력 경로 설정
# 파라미터 기록
# 로그 기록 설정
# ---
# 모델 경로 설정
# 차트 데이터, 학습 데이터 준비
# 최소/최대 투자 단위 설정
# 공통 파라미터 설정
# ---
# 강화학습 시작

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3'], default='v3')
    parser.add_argument(
        '--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'])
    parser.add_argument(
        '--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--output_name', default=settings.get_time_str())
    parser.add_argument('--value_network_name')
    parser.add_argument('--policy_network_name')
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--start_date', default='20200101')
    parser.add_argument('--end_date', default='20201231')
    args = parser.parse_args()

    # 출력 경로 설정
    output_path = os.path.join(
        settings.BASE_DIR, 'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))  # vars -> 딕셔너리

    # 디버깅
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR, CRITICAL 5단계

    # 디버그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함. 디버그설정을 먼저 해야 해댱 클래스에도 이 설정이 적용되기 때문
    from agent import Agent
    from learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(
            settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(output_path, '{}_{}_{}_value.h5'.format(
            args.output_name, args.rl_method, args.net))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(
            settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(output_path, '{}_{}_{}_policy.h5'.format(
            args.output_name, args.rl_method, args.net))

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method,
                         'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                         'balance': args.balance, 'num_epoches': args.num_epoches,
                         'discount_factor': args.discount_factor, 'start_epsilon': args.start_epsilon,
                         'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                                  'chart_data': chart_data,
                                  'training_data': training_data,
                                  'min_trading_unit': min_trading_unit,
                                  'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params,
                                        'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params,
                                                   'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params,
                                                'value_network_path': value_network_path,
                                                'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                args.net = args.rl_method
                args.num_epoches = 1
                args.discount_factor = None
                args.start_epsilon = 1
                args.learning = False
                learner = ReinforcementLearner(**common_params)
            if learner is not None:
                learner.run(learning=args.learning)
                learner.save_models()
