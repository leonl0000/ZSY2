import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle

opening_moves = np.load(open('utils/openingMoves.npz', 'rb'))
parameters = pickle.load(open("Parameters_M1_100epochs.pkl", 'rb'))

header = 'const math = require(\'mathjs\');\n\nvar matrices = {\n  test : math.matrix([[5,6], [3, 4]]),\n}\n'
footer = '\nexport default matrices'

outfilename = 'Matrices.js'

def generateJS(outfilename = outfilename):
    with open(outfilename, 'w+') as f:
        print(header, file = f)
        print("matrices[\'{0}\'] = math.matrix({1});\n\n".format('openingMoves', opening_moves.tolist()), file = f)
        len = 0
        for key, val in parameters.items():
            print("matrices[\'{0}\'] = math.matrix({1});\n\n".format(key, val.tolist()), file = f)
            len += 1
        print("matrices[\'{0}\'] = {1};\n\n".format('len', len/2), file = f)
        print(footer, file = f)
