import zipfile
import pandas as pd
from os.path import join, basename
from common import INPUT_DIR


def format(source, target):

    archive = zipfile.ZipFile(source, 'r')
    df = pd.DataFrame()

    for ind, filename in enumerate(archive.namelist()):
        with archive.open(filename) as file:

            uid = basename(filename)
            uid = uid[:uid.find('.')]

            if len(uid) == 0:
                continue

            sentences = []
            for line in file.readlines():
                line = line.decode('utf-8')
                if len(line) > 0:
                    sentences.append(line)
            text = " ".join(sentences)
            text = text.replace('\n', " ")
            text = " ".join(text.split())

            df = df.append({
                "id": uid,
                "s_ind": 0,
                "t_ind": 1,
                "text": text},
                ignore_index=True)

    df.to_csv(target, sep='\t')


# C1 task data.
format(source=join(INPUT_DIR, '6C_test_files.zip'),
       target=join(INPUT_DIR, "sample-orig-test-c1.tsv.gz"))

# C2 task data.
format(source=join(INPUT_DIR, '6C2_explanations_public_data.zip'),
       target=join(INPUT_DIR, "sample-orig-test-c2.tsv.gz"))
