from   dotenv import find_dotenv, load_dotenv
import os
import sys
#
load_dotenv(find_dotenv())
PATH_P     = os.environ['PATH_P']

os.chdir(PATH_P)
sys.path.insert(0, PATH_P)


































