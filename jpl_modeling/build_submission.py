import os
import csv
import json

basedir = os.path.abspath(os.path.dirname(__file__))


lookup = {}
with open(os.path.join(basedir, "eval_predictions.csv"), "rU") as f:
	reader = csv.reader(f)
	for row in reader:
		if reader.line_num > 1:
			phone = int(row[1])
			lookup[phone] = float(row[151])


with open(os.path.join(basedir, "submission"), "w") as out:
	with open(os.path.join(basedir, "raw_data", "ht_evaluation_NOCLASS"), "rU") as f:
	# with open(os.path.join(basedir, "ht_training_all"), "rU") as f:
		reader = f.readlines()
		for row in reader:
			row = json.loads(row)
			phone = int(row["phone"][0])

			# ideal_cutoff = .77

			#results
			doc = {}
			doc["phone"] = phone

			doc["score"] = lookup[phone]
			# doc["doc_id"] = row["ad"]["doc_id"]
			doc["ad_id"] = row["ad"]["_id"]
			doc["url"] = row["ad"]["url"]
			# if doc["score"] > ideal_cutoff:
			# 	doc["class"] = "positive"
			# else:
			# 	doc["class"] = "negative"
			out.write(json.dumps(doc) + "\n")
				


