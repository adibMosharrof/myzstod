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


class SimpleTodActionAttributes(str, Enum):
    domain = "domain"
    action_type = "action_type"
    slot_name = "slot_name"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class SpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    end_target = "<|endtarget|>"

    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"

    begin_dsts = "<|begindsts|>"
    end_dsts = "<|enddsts|>"

    begin_dst = "<|begindst|>"
    end_dst = "<|enddst|>"

    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"

    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"

    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"

    begin_user_action = "<|beginuseraction|>"
    end_user_action = "<|enduseraction|>"
    sys_actions = "<|sysactions|>"

    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"

    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"

    prompt_dst = "<|promptdst|>"
    prompt_action = "<|promptaction|>"
    prompt_response = "<|promptresponse|>"

    pad_token = "<|pad|>"
    eos_token = "<|endoftext|>"
    bos_token = "<|startoftext|>"

    begin_schema = "<|beginschema|>"
    end_schema = "<|endschema|>"
    schema_name = "<|schemaname|>"
    schema_description = "<|schemadescription|>"

    begin_schema_intent = "<|beginschemaintent|>"
    end_schema_intent = "<|endschemaintent|>"
    intent_required_slots = "<|intentrequiredslots|>"
    intent_result_slots = "<|intentresultslots|>"
    intent_optional_slots = "<|intentoptionalslots|>"

    begin_schema_slot = "<|beginschemaslot|>"
    end_schema_slot = "<|endschemaslot|>"
    schema_slot_values = "<|schemaslotvalues|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class SimpleTodConstants(str, Enum):
    DELEXICALIZED = "_delexicalized"
    SLOT_VALUE_SEPARATOR = "->"
    DOMAIN_SLOT_SEPARATOR = "^"
    ITEM_SEPARATOR = "|"
    ACTION_VALUE_SEPARATOR = "~"
    VALUE_SEPARATOR = ACTION_VALUE_SEPARATOR
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


class DstcSystemActions(str, Enum):
    INFORM = "INFORM"
    REQUEST = "REQUEST"
    CONFIRM = "CONFIRM"
    OFFER = "OFFER"
    NOTIFY_SUCCESS = "NOTIFY_SUCCESS"
    NOTIFY_FAILURE = "NOTIFY_FAILURE"
    INFORM_COUNT = "INFORM_COUNT"
    OFFER_INTENT = "OFFER_INTENT"
    REQ_MORE = "REQ_MORE"
    GOODBYE = "GOODBYE"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ContrastiveConstrants(str, Enum):
    NLG = "nlg"
    USER_ACT = "user_act"


class ContextType(str, Enum):
    SHORT_REPR = "short_repr"
    DEFAULT = "default"
