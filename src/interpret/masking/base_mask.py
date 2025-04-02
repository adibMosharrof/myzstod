class BaseMask:

    def __init__(self, cfg):
        self.cfg = cfg

    def get_mask(self, activations, percent):
        raise NotImplementedError()
