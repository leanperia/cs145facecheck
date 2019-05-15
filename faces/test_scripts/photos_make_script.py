import os
dirs = os.listdir(".")
print ([name for name in dirs if os.path.isdir(name)])
