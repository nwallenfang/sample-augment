# TODO add python PATH here
import sys
import os
src_path = os.path.dirname(os.path.abspath(__file__))
stylegan_path = os.path.join(src_path, 'models/generator/stylegan2/')
print('path:' + stylegan_path)
sys.path.insert(0, stylegan_path)
sys.path.insert(0, src_path)