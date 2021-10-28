''' This runfile starts the multi-agent system model.
    Also, prints the values of the model arguments.
    After the model has run, some statistics will be printed.
'''
from utils.parse_args import parse_args
from trust.model import PDTModel

DATA_PATH = 'data/'


def run():
    model_args, run_args, file_name = parse_args(True)

    model = PDTModel(**model_args)

    model.run_model(**run_args)
    df = model.datacollector.get_model_vars_dataframe()

    print(df.describe())
    df.to_csv(DATA_PATH + file_name)


if __name__ == "__main__":
    run()
