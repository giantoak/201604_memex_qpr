"""
Flattens raw ad data (json) into csv
"""

import os
import csv
import json

basedir = os.path.abspath(os.path.dirname(__file__))

#################
# training
#################
input_file = "ht_training_all"
output_file = "flattened_training.csv"

#################
# evaluation
#################
# input_file = "ht_evaluation_NOCLASS"
# output_file = "flattened_eval.csv"



flattened_data = []

ads = []
with open(os.path.join(basedir, "raw_data", input_file), "r") as f:
	raw = f.readlines()
	for ad in raw:
		ads.append(json.loads(ad))


##################################################
# Get all possible extracted fields from ads
##################################################
extraction_fields = []
for ad in ads:
	try:
		for key in ad["ad"]["extractions"]:
			if not key in extraction_fields:
				extraction_fields.append(key)
	except:
		print ad

#header row
flattened_data.append(["phone","class","_id"] + extraction_fields)

##################################################
# Put extractions into csv
##################################################
for ad in ads:
	features = []
	if "_id" in ad["ad"] and "class" in ad:
		features.extend([ad["phone"], ad["class"], ad["ad"]["_id"]]) 
	elif "_id" in ad["ad"] and not "class" in ad:
		features.extend([ad["phone"], "", ad["ad"]["_id"]]) # if don't have class (eval data) add blank
	else:
		features.extend([ad["phone"], ad["class"], ""]) 
		print ad # 5 missing ads
		print "=================="

	for extraction in extraction_fields:
		if "extractions" in ad["ad"] and extraction in ad["ad"]["extractions"]:
			features.append(ad["ad"]["extractions"][extraction]['results'])
		else:
			features.append("")

	flattened_data.append(features)
							

with open(os.path.join(basedir, output_file), "w") as out:
	writer = csv.writer(out)
	writer.writerows(flattened_data)

