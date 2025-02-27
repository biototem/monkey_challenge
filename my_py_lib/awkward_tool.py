import io
import awkward as ak
import csv


def load_ak_from_csv(csv_file):
    if isinstance(csv_file, str):
        csv_file = open(csv_file, 'r', encoding='utf8')

    assert isinstance(csv_file, io.FileIO)
    reader = csv.reader(csv_file)
    rows = []
    for row in reader:
        rows.append(row)

    arr = ak.Array(rows)
    return arr
