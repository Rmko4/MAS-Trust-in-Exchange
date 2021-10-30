import argparse
from typing import List

from utils.comp_range import Range

run_keys = ['T_onset', 'T_record']
save_keys = ['save_filename']


def pop_keys(dict: dict, keys: List[str]):
    items = {k: dict.pop(k) for k in list(dict.keys()) if k in keys}
    return items


def parse_args(print_args=False):
    parser = argparse.ArgumentParser(description='MAS for trust in exchange')
    parser.add_argument('-a', '--agent-class', dest='AgentClass', default='MSAgent',
                        choices=['MSAgent', 'WHAgent', 'RLAgent', 'GossipAgent', 'RLGossipAgent'])
    parser.add_argument('-m', '--mobility-rate', default=0.2,
                        type=float, choices=[Range(0.0, 1.0)])
    parser.add_argument('-N', '--number-of-agents', default=1000,
                        type=int, choices=[Range(0, 10000)])
    parser.add_argument('-n', '--neighbourhood-size',
                        default=30, type=int, choices=[Range(0, 10000)])

    parser.add_argument('-l', '--learning-rate', default=0.02,
                        type=float, choices=[Range(0.0, 1.0)], help='Only for RLAgent')
    parser.add_argument('-r', '--relative-reward', default=False,
                        type=bool, choices=[True, False], help='Only for RLAgent')
    parser.add_argument('-ms', '--memory-size', default=25,
                        type=bool, choices=[Range(0, 10000)], help='Only for GossipAgent')

    parser.add_argument('-t1', '--T_onset', default='100',
                        type=int, choices=[Range(0, int(1e6))])
    parser.add_argument('-t2', '--T_record', default='1000',
                        type=int, choices=[Range(1, int(1e6))])
    parser.add_argument('--save-filename', default='data.csv',
                        help='Saves to /m_SAVE-FILENAME and /a_SAVE-FILENAME')

    args = parser.parse_args()

    if args.neighbourhood_size > args.number_of_agents:
        raise ValueError(
            f'neighbourhood-size={args.neighbourhood_size} is larger than number-of-agents={args.number_of_agents}')
    if args.AgentClass not in ['RLAgent', 'RLGossipAgent']:
        del args.learning_rate
        del args.relative_reward
    if args.AgentClass not in ['GossipAgent', 'RLGossipAgent']:
        del args.memory_size

    kwargs = vars(args)

    run_args = pop_keys(kwargs, run_keys)
    save_filename = pop_keys(kwargs, save_keys)[save_keys[0]]

    if print_args:
        print("Model params: " + str(kwargs))
        print("Run params: " + str(run_args))

    return kwargs, run_args, save_filename
