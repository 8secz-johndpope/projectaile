from projectaile.utils import PIPELINE, CALLABLE


class PREPROCESS(CALLABLE):
    def __init__(self, apply_on_targets, **params):
        super(PREPROCESS, self).__init__(apply_on_targets, **params)


class PREPROCESSOR(PIPELINE):
    def __init__(self, config):
        super(PREPROCESSOR, self).__init__(config)