from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import json

from dotmap import DotMap


@dataclass
class Log:
    turn_id: int
    user_utterance: str
    system_response: str
    dialog_history: str
    original_user_side_information: dict
    original_system_side_information: dict
    dst: str
    dst_accumulated: str

    def __init__(self, item):
        self.turn_id = item["turn id"]
        self.user_utterance = item["user utterance"]
        self.system_response = item["system response"]
        self.dialog_history = item["dialog history"]
        self.original_user_side_information = DotMap(
            json.loads(item["original user side information"])
        )
        self.original_system_side_information = DotMap(
            json.loads(item["original system side information"])
        )
        self.dst = item["dst"]
        self.dst_accumulated = item["dst accumulated"]


@dataclass
class ExternalKnowledgeNonFlat:
    metadata: dict
    slots_and_values: dict
    intents: dict


@dataclass
class DsDialog:
    original_dialog_id: str
    dialog_index: int
    original_dialog_info: dict
    log: List[Log]
    # external_knowledge_non_flat: ExternalKnowledgeNonFlat
    external_knowledge_non_flat: str
    external_knowledge: str
    intent_knowledge: str
    prompt: List[str]
    services: Optional[list[str]] = None

    def __init__(self, item: dict):
        self.original_dialog_id = item["original dialog id"]
        self.dialog_index = item["dialog index"]
        self.original_dialog_info = item["original dialog info"]
        self.log = [Log(log) for log in item["log"]]
        self.external_knowledge_non_flat = item["external knowledge non-flat"]

        self.external_knowledge = item["external knowledge"]
        self.intent_knowledge = item["intent knowledge"]
        self.prompt = item["prompt"]
