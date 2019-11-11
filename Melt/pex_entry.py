
import chain
import common
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


summary = {}
chain.examine(sys.argv[1], summary)
print(common.jsdump(summary))
