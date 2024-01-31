import warnings
import argparse
import os
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conda', action='store_true')
    parser.add_argument('--pip', action='store_false', dest='conda')
    args = parser.parse_args()
    # Install required library for this project
    if args.conda is False:
        os.system('pip install -r requirements.txt')
    else:
        os.system('conda install --file requirements.txt')
