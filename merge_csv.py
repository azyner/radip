#!/usr/bin/env python
import fileinput

import os
import os.path

def merge_csvs(csv_name):
    # Collect all files
    file_list =[]
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(csv_name)]:
            file_list.append(os.path.join(dirpath, filename))
    header_over = False
    with open(csv_name, "w") as outfile:
        for line in fileinput.input(file_list):
            if fileinput.filelineno() == 1 and not header_over:
                outfile.write(line)
                header_over = True
            elif fileinput.filelineno() != 1:
                outfile.write(line)

merge_csvs("hyper.csv")
print "Finished merging hyper.csv"
merge_csvs("best.csv")
print "Finished merging best.csv"
