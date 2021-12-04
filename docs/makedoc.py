import os


if __name__ == "__main__":
    os.system("pdoc --html --force --config latex_math=True --output-dir . ../src")

