from enum import Enum


class Steps(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class TestSettings(str, Enum):
    SEEN = "seen"
    UNSEEN = "unseen"
    ALL = "all"
    CUSTOM = "custom"


class DstcDomains(list[str], Enum):
    SEEN = [
        "Banks",
        "Buses",
        "Calendar",
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
    UNSEEN = [
        "Alarm",
        "Messaging",
        "Payment",
        "Train",
    ]
    ALL = SEEN + UNSEEN


class Speaker(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class SpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    end_target = "<|endtarget|>"

    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"

    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"

    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"

    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"

    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"

    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"

    prompt_intent = "<|promptintent|>"
    prompt_requested_slots = "<|promptrequestedslots|>"
    prompt_belief = "<|promptbelief|>"
    prompt_action = "<|promptaction|>"
    prompt_response = "<|promptresponse|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class TokenizerTokens(str, Enum):
    pad_token = "<|pad|>"
    eos_token = "<|endoftext|>"
    bos_token = "<|startoftext|>"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.__str__()


class SimpleTodConstants(str, Enum):
    DELEXICALIZED = "_delexicalized"
    SLOT_VALUE_SEPARATOR = "->"
    DOMAIN_SLOT_SEPARATOR = "_"
    ITEM_SEPARATOR = "|"
    ACTION_VALUE_SEPARATOR = "<-"
    NEW_LINES = "\n\n"
    ACTION_TYPE_INFORM = "INFORM"
    ACTION_TYPE_INFORM_COUNT = "INFORM_COUNT"


class GoalMetricConfigType(str, Enum):
    ACTION = "action"
    BELIEF = "belief"

    def __repr__(self) -> str:
        return self.value


class SpecialPredictions(str, Enum):
    DUMMY = "DUMMY"
class TodMetricsEnum(str, Enum):
    BELIEF = "belief"
    ACTION = "action"
    INFORM = "inform"
    REQUESTED_SLOTS = "requested_slots"
    SUCCESS = "success"