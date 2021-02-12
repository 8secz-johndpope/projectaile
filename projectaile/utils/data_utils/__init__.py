from .extractors import *
from .feeder_utils import *

extractors = {
    'csv' : csv_parser,
    'dir' : dir_parser
}