from ..pipeline import PIPELINE
from ..callable import CALLABLE


class PREPROCESS(CALLABLE):
    def __init__(self, params):
        super(PREPROCESS, self).__init__(params)
        
    def __call__(self):
        pass


class PREPROCESSOR(PIPELINE):
    def __init__(self, config):
        super(PREPROCESSOR, self).__init__(config)