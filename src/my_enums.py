from enum import Enum
from typing import Optional


class TrainingStage(str, Enum):
    TRAIN = "train"
    PRETRAIN = "pretrain"


class Steps(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @classmethod
    def get_index(self, step_text):
        return Steps.list().index(step_text.lower())

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class Speaker(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class ZsTodActionAttributes(str, Enum):
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
    system = "<SYSTEM>"
    user = "<USER>"
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
    possible_values = "<|possiblevalues|>"
    end_possible_values = "<|endpossiblevalues|>"

    begin_schema_slot = "<|beginschemaslot|>"
    end_schema_slot = "<|endschemaslot|>"
    schema_slot_values = "<|schemaslotvalues|>"

    begin_service_results = "<|beginserviceresults|>"
    end_service_results = "<|endserviceresults|>"

    slot_value_separator = "<|value|>"
    domain_slot_separator = "<|domain|>"
    item_separator = "<|item|>"
    action_value_separator = "<|actionvalue|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ZsTodConstants(str, Enum):
    DELEXICALIZED = "_delexicalized"
    # SLOT_VALUE_SEPARATOR = "->"
    # DOMAIN_SLOT_SEPARATOR = "^"
    # ITEM_SEPARATOR = "|"
    # ACTION_VALUE_SEPARATOR = "~"
    SLOT_VALUE_SEPARATOR = SpecialTokens.slot_value_separator.value
    DOMAIN_SLOT_SEPARATOR = SpecialTokens.domain_slot_separator.value
    ITEM_SEPARATOR = SpecialTokens.item_separator.value
    ACTION_VALUE_SEPARATOR = SpecialTokens.action_value_separator.value
    VALUE_SEPARATOR = ACTION_VALUE_SEPARATOR
    NEW_LINES = "\n\n"
    ACTION_TYPE_INFORM = "INFORM"
    ACTION_TYPE_INFORM_COUNT = "INFORM_COUNT"


class GoalMetricConfigType(str, Enum):
    ACTION = "act"
    BELIEF = "goal"
    USER_ACTION = "u_act"

    def __repr__(self) -> str:
        return self.value


class SpecialPredictions(str, Enum):
    DUMMY = "DUMMY"


class TodMetricsEnum(str, Enum):
    BELIEF = "belief"
    ACTION = "action"
    USER_ACTION = "user_action"
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


class ContrastiveConstants(str, Enum):
    NLG = "nlg"
    USER_ACT = "user_act"
    LAST_UTTERANCE = "last_utt"


class ContextType(str, Enum):
    SHORT_REPR = "short_repr"
    DEFAULT = "default"
    NLG = "nlg"
    NLG_API_CALL = "nlg_api_call"
    KETOD_API_CALL = "ketod_api_call"
    KETOD_GPT_API_CALL = "ketod_gpt_api_call"
    BITOD = "bitod"
    BITOD_GPT = "bitod_gpt"
    CHATGPT = "chatgpt"
    GPT_API_CALL = "gpt_api_call"
    GPT_CROSS = "gpt_cross"
    GPT_PSEUDO_LABELS = "gpt_pseudo_labels"

    @classmethod
    def ketod_contexts(cls):
        return [cls.KETOD_API_CALL.value, cls.KETOD_GPT_API_CALL.value]

    @classmethod
    def bitod_contexts(cls):
        return [cls.BITOD.value, cls.BITOD_GPT.value]

    @classmethod
    def dstc_contexts(cls):
        return ContextType.list() - cls.ketod_contexts() - cls.bitod_contexts()

    @classmethod
    def list(cls):
        return [c for c in cls]


class ResponseMetricType(str, Enum):
    BLEU = "bleu"
    ROUGE = "rouge"


class MultiHeadName(str, Enum):
    DSTS = "dsts"
    SYSTEM_ACTIONS = "system_actions"
    NLG = "nlg"


class MultiTaskNames(str, Enum):
    DSTS = "dsts"
    ACTIONS = "actions"
    NLG = "nlg"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get_multi_task_names(
        self, tasks: Optional[list[str]]
    ) -> list["MultiTaskNames"]:
        if tasks is None:
            return self.list()
        try:
            multi_tasks = [self(mt) for mt in tasks]
        except:
            mt_values = ",".join([mt.value for mt in self])
            raise ValueError(
                f"multi_tasks must be one of the following {mt_values}, got {tasks}"
            )
        return multi_tasks

    @classmethod
    def list(cls):
        return [c for c in cls]


class TurnRowType(int, Enum):
    RESPONSE = 0
    API_CALL = 1
    KE_QUERY = 2
