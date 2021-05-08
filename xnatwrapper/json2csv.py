#!/usr/bin/env python3
import json,csv,os,glob

subject = os.environ["SUBJ"]
ses = os.environ["SES"]
outdir = os.environ["outdir"]

#Import json file
for folder in glob.glob(subject+'/'+ses+'/*/'):
	for json_file in glob.glob(folder+'/*.json'):
		with open(json_file,'r') as file:
			json_data=json.load(file);

		json_data.pop('bids_meta','provenance');

		#file_name = os.path.basename(json_file)

		csv_file = open(json_file.replace('.json','.csv'),'w');

		write=csv.writer(csv_file);

		write.writerow(json_data.keys());
		write.writerow(json_data.values());

		csv_file.close();
