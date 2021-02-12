from .settings import CONFIG
from .data import FEEDER
from .data import LOADER
from .models import MODEL

import os
from shutil import copytree


'''
    create_architecture : creates the ProjectAile architecture
'''
def create_architecture(params):
    script_path = os.path.realpath(__file__)
    script_path = script_path.replace('\\', '/') if '\\' in script_path else script_path
    copytree(script_path+'/architecture/', params['folder_dir'])