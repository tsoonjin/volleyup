#!/usr/bin/env python
import argparse
from sklearn.externals import joblib


def main():
    parser = argparse.ArgumentParser(description='Convert between pickle protocol versions.')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--version', default=2, type=int, help='The pickle protocol version to write the OUTPUT_FILE.')
    args = parser.parse_args()

    data = joblib.load(args.input_file)

    joblib.dump(data, args.output_file, args.version)

    print("Done!")

if __name__ == "__main__":
    main()
