#json2csv
import sys
import json
import csv

with open(sys.argv[1]) as fp:
    j = json.load( fp )
    j = j['comments']

fields = []
for comment in j:
    for k in comment.keys():
        if k not in fields:
            fields.append(k)
writer = csv.DictWriter(sys.stdout, fieldnames=fields, quoting=csv.QUOTE_NONNUMERIC)
writer.writeheader()
for comment in j:
    writer.writerow(comment)
