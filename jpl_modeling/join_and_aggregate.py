"""
Aggregates data files from different groups 
Rolls up data from the ad level to the phone number level
Outputs csv for modeling
"""

import os
import csv
import json
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

basedir = os.path.abspath(os.path.dirname(__file__))

#################
# training
#################
# eval_data = False
# giant_oak_file = "data_to_use_by_ad_v3_with_exp_imgs.csv"
# naive_bayes_file = "nb_training"
# multimedia_file = "mm_training.csv"
# lattice_file = "cp1_train.json"
# raw_file = "flattened_training.csv"
# output_file = "modeling_training.csv"


#################
# evaluation
#################
eval_data = True
giant_oak_file = "data_to_use_by_ad_test_with_exp_imgs.csv"
naive_bayes_file = "nb_eval.csv"
multimedia_file = "mm_eval.jl"
lattice_file = "cp1_eval.json"
raw_file = "flattened_eval.csv"
output_file = "modeling_eval.csv"

##############################################################
# Read in Giant Oak csv as dictionary 
# ----------------------------------------------------
# structure: 
# { <phone_num> : { <field> : { <value> : count of value } } }
##############################################################
giant_oak_data = {}
# field name : column number 
fields = {
	"class" : 0,
	"flags" : 5,
	"age" : 6,
	"metros" : 7,
	"ids" : 10,
	"price_per_min" : 11,
	"diseases" : 12,
	"year" : 13,
	"cases" : 14,
	"rate" : 15,
	"property" : 17,
	"rape" : 18,
	"violent" : 19,
	"images" : 60,
	"simimages_ads" : 61,
	"simimages" : 62
}

if eval_data: 
	for x in fields.keys():
		if x != "class": #not present in eval data
			fields[x] = fields[x] - 1
		else:
			fields.pop(x, None)
			pass


with open(os.path.join(basedir, "other_groups_data", "giant_oak", giant_oak_file), "rU") as f:
	reader = csv.reader(f)
	for row in reader:
		phone = row[1].replace("(","").replace(")","").replace("-","").replace(" ","") # use digits only for phone nums
		if not phone in giant_oak_data:
			giant_oak_data[phone] = {}
			for key, value in fields.iteritems():
				giant_oak_data[phone][key] = {}
		else:
			for key, value in fields.iteritems():
				if row[value] != "":
					if not row[value] in giant_oak_data[phone][key]:
						giant_oak_data[phone][key][row[value]] = 1
					else:
						giant_oak_data[phone][key][row[value]] += 1


########################################################################
# Read in predicted classes from naive bayes model to be used as feature
########################################################################

nb_lookup = {}
with open(os.path.join(basedir, "other_groups_data", "jpl_naive_bayes", naive_bayes_file), "rU") as k:
	reader = csv.reader(k)
	for row in reader:
		nb_lookup[row[0]] = row[1]


############################################################################
# Read in predicted class scores from Kitware MM model to be used as feature
############################################################################
mm_lookup = {}
with open(os.path.join(basedir, "other_groups_data", "kitware", multimedia_file), "rU") as k:
	#Different formats provided by kitware
	if "training" in multimedia_file: 
		reader = csv.reader(k)
		for row in reader:
			mm_lookup[row[0]] = row[1]
	elif "eval" in multimedia_file:
		reader = k.readlines()
		for row in reader:
			row = json.loads(row)
			mm_lookup[row["phone"]] = row["score"]


#######################################################
# Read in lattice data
# ----------------------------------------------------
# structure: 
# { <phone_num> : [ { <ad1 json> }, { <ad2 json> } ] }
#######################################################

lattice_data = {}
flags = [u'Foreign Providers', u'Juvenile', u'Multiple Girls', u'Traveling', u'URL Embedding', 
u'Business Addresses', u'Massage Parlor', u'Accepts Walk-ins', u'Hotel', u'Accepts Credit Cards', 
u'Risky Services', u'Agency']

with open(os.path.join(basedir, "other_groups_data", "lattice", lattice_file), "r") as f:
	reader = f.readlines()
	for row in reader:
		row = json.loads(row)
		for phone in row["phones"]:
			phone = phone.replace("(","").replace(")","").replace("-","").replace(" ","")
			if not phone in row:
				lattice_data[phone] = [row]
			else:
				lattice_data[phone].append(row)


###############################################################
# Join and aggregate data to create features at phone num level
# -------------------------------------------------------------
# structure: 
# aggregated = { <phone_num> : { <feature (field)> : value } }
###############################################################

aggregated = {} 

fields = ["phone","ad_count","mm_score","unique_flags_cnt", "flags_cnt", "match",
"unique_age_cnt","age_range","min_age","max_age","average_age", "most_common_age",
"unique_price_per_min_cnt", "price_per_min_range", "max_price_per_min","min_price_per_min", "average_price_per_min", "most_common_price_per_min",
"unique_year_cnt", "year_range", "max_year","min_year", "average_year", "most_common_year",
"unique_rate_cnt", "rate_range", "max_rate","min_rate", "average_rate", "most_common_rate",
"unique_images_cnt", "images_range", "max_images","min_images", "average_images", "most_common_images",
"unique_simimages_cnt", "simimages_range", "max_simimages","min_simimages", "average_simimages", "most_common_simimages",
"unique_simimages_ads_cnt", "simimages_ads_range", "max_simimages_ads","min_simimages_ads", "average_simimages_ads", "most_common_simimages_ads",
"unique_cases_cnt", "cases_range", "max_cases","min_cases", "average_cases", "most_common_cases",
"diseases_cnt","foreign_providers","juvenile", "multiple_girls", "traveling","url_embedding",
"bus_address","massage_parlor","walk_ins","hotel","cc","risky","agency","primary_domain",
"num_metros", "num_cities"]

if eval_data:
	fields.remove("match")

numeric_fields = ["age", "price_per_min", "year", "rate", "images", "simimages_ads", "simimages", "cases"]

#don't know where these came from (saw in slack)
keywords = ["khaleah","text","sixsix","drama","legends","hey","pls","fit","convo","like","constance",
"blck","deal","accurate","lookn","total","tex","jessica","aa","petite","shanell","deal","gorgious",
"discreet","thr","experiance","jessica","blonde","sneak","packageguaranteed","honey","face","dita",
"pretty","fit","wanting","searching","dominicanspent","kelli","charisma","street","erotic","loving",
"guaranteed","ddi","tender","karmen","badgirl","skyy","ebony","playmates","looking","hang","perfect",
"nights","time","enjoys","unique","text","freaky","drama","unforgettable","butt","freak","tender",
"things","world","college","banks","king","affiliation","sex","classy","passion","spcls","easy", 
"keyword_cnt","nb_class"]

fields = fields + keywords # use individual keyword counts as features

missing_phones = 0 #track number of phones not in Giant Oak data

with open(os.path.join(basedir, raw_file), "r") as f: # file is output of initial flattening script
	reader = csv.reader(f)
	for row in reader:
		if reader.line_num > 1: #skip header row 

			for phone in eval(row[0]): 
				phone = str(int(phone))

				# initialize key if not in dict
				#===========================================================
				if not phone in aggregated:
					aggregated[phone] = {}
					aggregated[phone]["ad_count"] = 0
					# aggregated[phone]["doc_ids"] = []


				# keep track of number of ads available for phone number
				#===========================================================
				aggregated[phone]["ad_count"] += 1


				# multimedia score (Kitware)
				#===========================================================
				aggregated[phone]["mm_score"] = mm_lookup[phone] 


				# class - recode to T/F			
				#===========================================================	
				if row[1] == "positive":
					aggregated[phone]["match"] = "TRUE"
				else:
					aggregated[phone]["match"] = "FALSE"


				# Giant Oak data
				#===========================================================
				if phone in giant_oak_data:

					#number of total ads with flags (flag column is populated)
					aggregated[phone]["unique_flags_cnt"] = len(set(giant_oak_data[phone]["flags"].keys()))
					total_flags = 0
					for x in giant_oak_data[phone]["flags"].keys():
						total_flags += giant_oak_data[phone]["flags"][x]
					aggregated[phone]["flags_cnt"] = total_flags
					
					#initialize blanks for numeric fields
					for key in numeric_fields:
						aggregated[phone]["unique_" + key + "_cnt"] = ""
						aggregated[phone][key + "_range"] = ""
						aggregated[phone]["min_" + key] = ""
						aggregated[phone]["max_" + key] = ""
						aggregated[phone]["average_" + key] = ""
						aggregated[phone]["most_common_" + key] = ""

					# for all numeric fields, calculate number of unique values, range, min, max, average
					for key in numeric_fields:
						if len(giant_oak_data[phone][key]) > 0:
							vals = 	map(float,giant_oak_data[phone][key].keys())
							aggregated[phone]["unique_" + key + "_cnt"] = len(vals)
							aggregated[phone][key + "_range"] = max(vals) - min(vals)
							aggregated[phone]["min_" + key] = min(vals)
							aggregated[phone]["max_" + key] = min(vals)
							num_vals = 0
							_sum = 0
							max_key = 0
							for key2,value2 in giant_oak_data[phone][key].iteritems():
								num_vals += value2
								_sum += (float(key2) * value2)
								if value2 > max_key:
									max_key = value2
							aggregated[phone]["average_" + key] = _sum / num_vals
							aggregated[phone]["most_common_" + key] = max_key

					#number of ads with diseases
					aggregated[phone]["diseases_cnt"] = len(giant_oak_data[phone]["diseases"])

				else:
					missing_phones += 1


				#Lattice data features
				#========================================
				if phone in lattice_data:
					aggregated[phone]["primary_domain"] = []
					aggregated[phone]["num_metros"] = []
					aggregated[phone]["num_cities"] = []

					for ad in range(0,len(lattice_data[phone])):

						#Keep counts for each individual flag
						if lattice_data[phone][ad]["flags"] and len(lattice_data[phone][ad]["flags"]) > 0:
							for flag in lattice_data[phone][ad]["flags"]:
								# foreign providers is first flag in "fields" list (use index to find appropriate key)
								if not fields[fields.index("foreign_providers") + flags.index(flag)] in aggregated[phone]:
									aggregated[phone][fields[fields.index("foreign_providers") + flags.index(flag)]] = 1
								else:
									aggregated[phone][fields[fields.index("foreign_providers") + flags.index(flag)]] += 1

						# create list of domains associated with a phone num
						if "domain" in lattice_data[phone][ad]:
							aggregated[phone]["primary_domain"].append(lattice_data[phone][ad]["domain"])

						# create list of domains metropolitan ares associated with a phone num
						
						if "metropolitan_areas" in lattice_data[phone][ad] and not lattice_data[phone][ad]["metropolitan_areas"] == None:	
							aggregated[phone]["num_metros"].extend(lattice_data[phone][ad]["metropolitan_areas"])

							
						# create list of domains cities associated with a phone num
						if lattice_data[phone][ad]["cities"] and len(lattice_data[phone][ad]["cities"]) > 0:
							aggregated[phone]["num_cities"].extend(lattice_data[phone][ad]["cities"])


					for x in range(0, len(flags)):
						if not fields[fields.index("foreign_providers") + x] in aggregated[phone]:
							aggregated[phone][fields[fields.index("foreign_providers") + x]] = 0

					aggregated[phone]["primary_domain"] = Counter(aggregated[phone]["primary_domain"]).most_common()[0][0]
					# count number of unique metropolitan areas / cities associated with phone num
					aggregated[phone]["num_metros"] = len(set(aggregated[phone]["num_metros"]))
					aggregated[phone]["num_cities"] = len(set(aggregated[phone]["num_cities"]))


				# Count of each keyword found in ad texts for given phone num
				# ============================================================
				keyword_cnt = 0
				for word in keywords:
					if word in row[13]:
						aggregated[phone][word] = row[13].count(word)
						keyword_cnt += 1
					else:
						aggregated[phone][word] = 0
				aggregated[phone]["keyword_cnt"] = keyword_cnt	


###############################################
# Write out values for each field if they exist 
###############################################
with open(os.path.join(basedir, output_file), "w") as out:
	writer = csv.writer(out)
	writer.writerow(fields)
	for key,value in aggregated.iteritems():
		# row = [key]
		row = []
		row.append(key)
		for field in fields:
			if field != "phone":
				if field in aggregated[key]:
					row.append(aggregated[key][field])
				else:
					row.append("")
		writer.writerow(row)




