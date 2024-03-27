from collections import Counter, defaultdict
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("./src"))
from dotmap import DotMap
import pandas as pd
from tqdm import tqdm

from multi_head.mh_dataclasses import MultiHeadDictFactory

from sgd_dstc8_data_model.dstc_dataclasses import DstcDialog, DstcSchema
from my_enums import Steps
import utils
import pandas as pd
from transformers import AutoTokenizer
import dstc.dstc_utils as dstc_utils

# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.abspath(myPath + "/../"))


class MhLengths:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        data = []
        col_names = None
        for step, num_dialog in zip(Steps, self.cfg.num_dialogs):
            file_name = self.cfg.csv_file_name.replace("numdialogs", str(num_dialog))
            file_path = self.cfg.processed_data_root / step.value / file_name
            row, col_names = utils.read_csv(file_path)
            data.extend(row)
        df = pd.DataFrame(data)
        # df = df[range(4, 10)]
        df.columns = col_names
        tok = dstc_utils.get_tokenizer()

        counts = dict.fromkeys(col_names, [])
        for head in col_names:
            df_col = df[head]
            counts[head] = df_col.apply(lambda x: self.my_tokenize(tok, x)).to_numpy()
            # counts[head] = [self.my_tokenize(tok, x) for x in df_col]
        max_counts = dict.fromkeys(col_names, 0)
        for head in col_names:
            max_counts[head] = counts[head].max()
        print(max_counts)
        a = 1

    def my_tokenize(self, tok, data):
        if not data:
            return 0
        return len(tok.encode(data))


if __name__ == "__main__":
    mhl = MhLengths(
        DotMap(
            processed_data_root=Path("processed_data/simple_tod"),
            num_dialogs=[127, 20, 34],
            csv_file_name="v0_context_type_short_repr_multi_head_True_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_True_turns_26_service_results_True_dialogs_numdialogs_delexicalize_False_domain_setting_SEEN_train_domains_1.0.csv",
        )
    )
    mhl.run()
