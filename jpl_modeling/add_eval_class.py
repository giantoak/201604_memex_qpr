import os
import csv
import json

basedir = os.path.abspath(os.path.dirname(__file__))

phones = []
answer_lookup = {}
with open(os.path.join(basedir, "raw_data", "ht_evaluation_UPDATED"), "rU") as f:
	# with open(os.path.join(basedir, "ht_training_all"), "rU") as f:
	reader = f.readlines()
	for row in reader:
		row = json.loads(row)

		for x in row["phone"]:
			phone = int(x)
			if not phone in phones:
				phones.append(phone)
			if row["class"] == "positive":
				answer_lookup[phone] = "TRUE"
			if row["class"] == "negative":
				answer_lookup[phone] = "FALSE"

lookup = {}
with open(os.path.join(basedir, "eval_predictions.csv"), "rU") as f:
	with open(os.path.join(basedir, "eval_predictions_with_class.csv"), "w") as f2:
		reader = csv.reader(f)
		writer = csv.writer(f2)
		for row in reader:
			if reader.line_num > 1:
				phone = int(row[1])
				row.append(answer_lookup[phone])
			else:
				row.append("match")
			writer.writerow(row)




				


