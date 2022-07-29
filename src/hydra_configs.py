class InferenceConfig:
    def __init__(
        self,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 100,
        max_token_len: int = 512,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        data_prep_out_root: str = "processed_data/simple_tod",
        num_test_dialogs: int = 1,
        delexicalize: bool = False,
        model: str = "outputs/2022-07-26/22-28-09/results/train/checkpoint-7067",
        model_name: str = "gpt2",
        device: str = "cuda",
        generate_max_len: int = 1024,
        domains: list[str] = None,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        test_settings: list[str] = None,
    ) -> None:
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 0.1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.max_token_len = max_token_len
        self.raw_data_root = raw_data_root
        self.project_root = project_root
        self.data_prep_out_root = data_prep_out_root
        self.num_test_dialogs = num_test_dialogs
        self.delexicalize = delexicalize
        self.model = model
        self.model_name = model_name
        self.device = device
        self.generate_max_len = generate_max_len
        self.domains = domains or [
            "Buses",
            "Events",
            "Flights",
            "Homes",
            "Hotels",
            "Media",
            "Movies",
            "Music",
            "RentalCars",
            "Restaurants",
            "RideSharing",
            "Services",
            "Travel",
            "Weather",
        ]
        self.test_settings = test_settings or ["seen"]
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
