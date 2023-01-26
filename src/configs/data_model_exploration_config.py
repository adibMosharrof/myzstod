from dstc.dstc_domains import DstcDomains
import dstc.dstc_utils as dstc_utils
from pathlib import Path

class DataModelExplorationConfig:
    def __init__(
        self,
        data_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        project_root: str = None,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        model_name: str = "gpt2",
        out_root: str = "model_exploration",
        num_turns: int = 10,
        domain_setting: str = "SEEN",
        overwrite: list[bool] = None,
        data_split_percent: list[float] = None,
        is_multi_task: bool = False,
        should_add_schema: bool = False,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.raw_data_root = self.project_root / raw_data_root
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.tokenizer = dstc_utils.get_tokenizer(model_name)
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.should_add_schema = should_add_schema
        self.overwrite = overwrite or [False, False, False]
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.domain_setting = domain_setting
        self.domains = DstcDomains[domain_setting.upper()].value

