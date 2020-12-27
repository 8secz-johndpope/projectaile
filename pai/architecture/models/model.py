'''
    Contains The Model Code

'''

import projectaile as pai


'''
    Model Definition
'''


class __MODEL_NAME__(pai.MODEL):
    '''
        Constructor
    '''

    def __init__(self):
        # Initialization. Creates projectAIle.MODEL Instance From The config.json File In The Settings Folder
        super(__MODEL_NAME__, self).__init__('../settings/config.json')

    # Main Method For You To Customize ( Define The Model Architecture And Return The Model Object )
    def compose_model(self):
        pass

    # You Can Add Any Additional Methods You Want, projectAIle.MODEL Does Provide Some Utility Methods, Check Out The Documentation.
