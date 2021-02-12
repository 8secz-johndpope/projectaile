class CALLABLE:
    def __init__(self, **kwargs):
        self.params = kwargs
        
    def __call__(self, inp):
        return