class PIPELINE:
    def __init__(self, process_type, config):
        self.config = config
        self.process_type = process_type
        self.processes = []
        for item in config:
            self.processes.append(self.get_method(item['type'])(**item['params']))
        
        
    def get_method(self, name):
        try:
            getattr(self.process_type, name)
        except Exception as e:
            print(f'Can not find {name} in {self.process_type}')
            
    def apply(self, inp):
        for process in self.processes:
            inp = process(inp)
        
        return inp