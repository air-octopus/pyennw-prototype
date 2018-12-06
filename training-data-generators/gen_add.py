# coding=utf-8

from training_data import TrainingData

import json
from random import Random

td = TrainingData(json_str="""
{
    "training_set_name": "add",
    "inputs": ["input_01", "input_02"],
    "outputs": ["add"],
    "combo_set_ratio": 0.8,
    "combo_set": [],
    "training_set": [],
    "testing_set": []
}
""")

r = Random()

def get_datum():
    while True:
        a = r.uniform(0, 10)
        b = r.uniform(0, 10)
        yield {"in": [a, b], "out": [a+b]}

datum_it = get_datum()

for i in range(1000):
    td._training_set.append(datum_it.__next__())

for i in range(100):
    td._testing_set.append(datum_it.__next__())

with open("../data/training-data.json", "w") as f:
    json.dump(td.json_all, f, indent=4)

pass
