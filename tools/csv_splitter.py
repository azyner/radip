#!/usr/bin/env python
import os
import argparse

# Derived from https://gist.github.com/jrivero/1085501#file-csv_splitter-py
parser = argparse.ArgumentParser(description='Split CSV into files of N lines')
parser.add_argument('--rows', type=int, nargs=1, default=4000000, help='Number of lines per file')
parser.add_argument('filename', type=str, nargs=1, help='csv filename')
args = parser.parse_args()

def split(filehandler, delimiter=',', row_limit=4000000,
    output_name_template='output_%02d.csv', output_path='.', keep_headers=True):
    """
    Splits a CSV file into multiple pieces.
    
    A quick bastardization of the Python CSV library.

    Arguments:

        `row_limit`: The number of rows you want in each output file. 10,000 by default.
        `output_name_template`: A %s-style template for the numbered output files.
        `output_path`: Where to stick the output files.
        `keep_headers`: Whether or not to print the headers in each output file.

    Example usage:
    
        >> from toolbox import csv_splitter;
        >> csv_splitter.split(open('/home/ben/input.csv', 'r'));
    
    """
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
         output_path,
         output_name_template  % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
               output_path,
               output_name_template  % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)

split(open(args.filename[0],'r'),row_limit=args.rows,output_name_template="split_"+args.filename[0][0:-4]+'_%02d.csv')