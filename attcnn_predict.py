from arenets.arekit.common.data_type import DataType
from arenets.context.configurations.cnn import CNNConfig
from arenets.core.writer.csv_writer import CsvContentWriter
from arenets.external.readers.pandas_csv_reader import PandasCsvReader
from arenets.quickstart.predict import predict
from arenets.enum_name_types import ModelNames
from common import INPUT_TERMS_COUNT, OUTPUT_DIR, INPUT_DIR
from provider.explanation import provide_explanation
from provider.prediction import LegalTask6C2PredictProvider


def modify_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_terms_per_context(INPUT_TERMS_COUNT)
    config.set_filters_count(600)


predict(input_data_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        labels_count=2,
        model_name=ModelNames.CNNAttSelfPZhou,
        bags_per_minibatch=32,
        reader=PandasCsvReader(),
        data_type=DataType.Test,
        predict_provider=LegalTask6C2PredictProvider(),
        predict_writer=CsvContentWriter(separator=',', write_title=True),
        modify_config_func=modify_config,
        unknown_term_index=1814)  # unknown word

provide_explanation(input_dir=INPUT_DIR,
                    model_name=ModelNames.CNNAttSelfPZhou.value,
                    input_terms_count=INPUT_TERMS_COUNT,
                    sample_type=DataType.Test,
                    output_bound=64,
                    sentence_window=10,
                    output_dir=OUTPUT_DIR,
                    reader=PandasCsvReader(),
                    extention="tsv.gz")
