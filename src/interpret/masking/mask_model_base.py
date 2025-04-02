class MaskModelBase:

    def mask(self, model, mask):
        model.transformer.mask_m_layer = mask
        return model
