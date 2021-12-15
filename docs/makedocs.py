import os
from os.path import sep

if __name__ == "__main__":
    os.system("pdoc --html --config latex_math=True --output-dir . .." + sep + "src")

    assert(os.getcwd().split(sep)[-1] == "docs")
    # TODO always navigate in docs no matter the cwd and do CI

    for file in os.listdir("src"):
        os.rename(os.getcwd() + sep + "src" + sep + file, os.getcwd() + sep + file)

    os.rmdir("src")

