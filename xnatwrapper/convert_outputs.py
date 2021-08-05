#!/usr/bin/env python

#	 4. Show a string that we'll pass in from the yaml,
#	  that would have the XNAT-specific project, subject, session, scan

import json,csv,os,glob
from fpdf import FPDF
from datetime import datetime



#Import json file and write to csv
for folder in glob.glob('sub-*/ses-*/*'):
	for json_file in glob.glob(folder+'/*.json'):
		with open(json_file,'r') as file:
			json_data=json.load(file);
		json_data.pop('bids_meta');
		json_data.pop('provenance');
		csv_file = open(json_file.replace('.json','.csv'),'w');
		write=csv.writer(csv_file);
		write.writerow(json_data.keys());
		write.writerow(json_data.values());
		csv_file.close();

#Initialize pdf
loc = glob.glob('sub-*/ses-*/*')
fldrs = loc[0].split('/')

pdf=FPDF()
pdf.add_page()
page_width = pdf.w - 2 * pdf.l_margin
pdf.set_font('Times','B',16.0)
pdf.cell(page_width,0.0,fldrs[0].capitalize() + ' ' + fldrs[1].capitalize() + ' Image Quality Metrics',align='C')
pdf.ln(8)

# Write xnat specific project info to header
pdf.set_font('Times','',14.0)
label_info = os.environ["label_info"]
if label_info:
	label_info = label_info.split(' ')
	pdf.cell(page_width,0.0,'XNAT Project: ' + label_info[0] + ' Subject: ' + label_info[1] + ' Session: ' +  label_info[2] + ' Scan: ' +  label_info[3],align='C')
else:
	pdf.cell(page_width,0.0,'No XNAT Label Provided',align='C')
pdf.ln(8)

# Write MRIQC version to header
version = os.environ["version"]

pdf.cell(page_width,0.0,'MRIQC Version ' + version,align='C')
pdf.ln(8)

# Get current time and write to header
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

pdf.cell(page_width,0.0,dt_string,align='C')
pdf.ln(12)

#Load in csv file and write to pdf
first = True
for folder in glob.glob('sub-*/ses-*/*'):
	for csv_file in glob.glob(folder+'/*.csv'):
		with open(csv_file, newline='') as f:

			if first:
				first = False
			else:
				pdf.add_page()

			tmp = csv_file.split('_')
			scan = tmp[-1].replace('.csv','')

			pdf.set_font('Times','B',14.0)
			pdf.cell(page_width,0.0,scan.capitalize())
			pdf.ln(6)

			reader = csv.reader(f)

			pdf.set_font('Times', '', 12)
			col_width = page_width/3
			th=pdf.font_size
			for row in zip(*reader): 
				pdf.cell(col_width, th, row[0],border=1)
				pdf.cell(col_width, th, row[1],border=1)
				pdf.ln(th)        

			#pdf.ln(20)
			#pdf.add_page()


# Finalize pdf
pdf.output(fldrs[0] + '_' + fldrs[1] + '_MRIQC_IQMs.pdf','F')
