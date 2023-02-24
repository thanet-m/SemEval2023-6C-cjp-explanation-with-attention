from arenets.arekit.common.data_type import DataType
from arenets.context.configurations.cnn import CNNConfig
from arenets.core.writer.csv_writer import CsvContentWriter
from arenets.external.readers.pandas_csv_reader import PandasCsvReader
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames
from common import INPUT_TERMS_COUNT, OUTPUT_DIR, INPUT_DIR
from provider.prediction import LegalTask6C1PredictProvider


def modify_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_terms_per_context(INPUT_TERMS_COUNT)
    config.set_filters_count(600)


predict(input_data_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        labels_count=2,
        model_name=ModelNames.CNN,
        bags_per_minibatch=32,
        reader=PandasCsvReader(),
        data_type=DataType.Test,
        predict_provider=LegalTask6C1PredictProvider(),
        predict_writer=CsvContentWriter(separator=',', write_title=True),
        modify_config_func=modify_config,
        unknown_term_index=1814)  # unknown word for a default embedding
