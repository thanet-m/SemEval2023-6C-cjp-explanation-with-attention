from arenets.context.configurations.cnn import CNNConfig
from arenets.external.readers.pandas_csv_reader import PandasCsvReader
from arenets.quickstart.train import train
from arenets.enum_name_types import ModelNames
from arenets.core.callback.train_limiter import TrainingLimiterCallback

from common import INPUT_TERMS_COUNT, INPUT_DIR


def modify_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_terms_per_context(INPUT_TERMS_COUNT)
    config.set_filters_count(600)


train(input_data_dir=INPUT_DIR,
      labels_count=2,
      model_name=ModelNames.CNNAttSelfPZhou,
      epochs_count=50,
      reader=PandasCsvReader(),
      bags_per_minibatch=32,
      learning_rate=0.01,
      modify_config_func=modify_config,
      callbacks=[
          TrainingLimiterCallback(train_acc_limit=0.99)
      ],
      unknown_term_index=1814)  # unknown word

