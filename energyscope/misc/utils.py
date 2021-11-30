import os


# TODO: comment
# Useful functions for printing in AMPL syntax #
def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
