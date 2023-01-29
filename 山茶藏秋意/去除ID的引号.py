import csv

f = open(r"D:\yiguohuang\people\code_people\bank-full_2.csv", 'rt')

try:

    for row in csv.reader(f, delimiter=' ', skipinitialspace=True):

        print('|'.join(row))

finally:

    f.close()
