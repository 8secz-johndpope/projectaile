class PIPELINE:
    def __init__(self, config, flow_method='pass'):
        self.config = config
        self.processes = []
        for item in config:
            self.processes.append(self.get_method(item['type'])(**item['params']))
    
    def get_method(self, type):
        return
    
    def apply(self, inp):
        for process in self.processes:
            inp = process(inp)
        
        return inp