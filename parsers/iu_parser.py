"""
Parses IU dataset and saves json containing information.
Language/report data needs to be further processed by CheXpert.

Usage:
python iu_parser.py

Note:
Assumes that indiana_projections.csv and indiana_reports.cv are in the same
directory as this python script.

Format of IU json:

{uid, frontal/lateral} - defines unique patient and image
{mesh, problems, indication, comparison, findings, impression} - language info
split - train, test or val
label - decided by language info & CheXpert
"""

import json
import pandas as pd
import numpy as np


reports = 'indiana_reports.csv'
projs = 'indiana_projections.csv'

TRAIN_PER = 80
VAL_PER = 10
TEST_PER = 10

# Patient info stored in a dictionary that mirrors above structure
patient_info_dict = {}

rdata = pd.read_csv(reports, sep=',')
pdata = pd.read_csv(projs, sep=',')
total_patients = len(rdata.index)

train_idxs = list(range(0, int(total_patients * 0.8)))
val_idxs = list(range(int(total_patients * 0.8), int(total_patients * 0.9)))
test_idxs = list(range(int(total_patients * 0.9), total_patients))

# Iterate through report and gather data
for index, row in rdata.iterrows():
	# Patient ID
	uid = row['uid']
	
	# Language data
	mesh = row['MeSH']
	problems = row['Problems']
	indication = row['indication']
	comparison = row['comparison']
	findings = row['findings']
	impression = row['impression']

	# Get unique patient key and image filepath
	uid_idxs = pdata.index[pdata['uid'] == uid].tolist()
	for uii in uid_idxs:
		proj_type = pdata.loc[uii, 'projection']
		key = '{}_{}'.format(uid, proj_type)
		img_path = pdata.loc[uii, 'filename']
		# Labels will be added later after CheXpert analysis
		if index in train_idxs:
			split = 'train'
		elif index in val_idxs:
			split = 'val'
		elif index in test_idxs:
			split = 'test'
		else:
			raise RuntimeError('Unable to determine split!')
		# Replace NaN values with empty strings
		patient_info = {
			'img_path': img_path,
			'mesh': mesh if not pd.isna(mesh) else "",
			'problems': problems if not pd.isna(problems) else "",
			'indication': indication if not pd.isna(indication) else "",
			'comparison': comparison if not pd.isna(comparison) else "",
			'findings': findings if not pd.isna(findings) else "",
			'impression': impression if not pd.isna(impression) else "",
			'split': split,
			'labels': [-1]
		}
		patient_info_dict[key] = patient_info

# Write dictionary to json
json_dict = {'root': patient_info_dict}
json_fname = 'iu_annotated.json'
with open(json_fname, 'w') as write_file:
	json.dump(json_dict, write_file, indent=4, sort_keys=False)