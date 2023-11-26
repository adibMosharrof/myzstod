from dataclasses import dataclass


@dataclass
class NlgTodTarget:
    response: str

    def __str__(self):
        return self.response
