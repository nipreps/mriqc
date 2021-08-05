#!/usr/bin/env python

<<<<<<< HEAD
=======
# To add:
#	Add outlog to pdf (seperate?)


>>>>>>> 5cdf1166ac37b5054341e7bcffbc9e777295a131
import json,csv,os,glob
from fpdf import FPDF

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
pdf.ln(15)

#Load in csv file and write to pdf
for folder in glob.glob('sub-*/ses-*/*'):
	for csv_file in glob.glob(folder+'/*.csv'):
		with open(csv_file, newline='') as f:

			tmp = csv_file.split('_')
			scan = tmp[-1].replace('.csv','')

			pdf.set_font('Times','B',14.0)
			pdf.cell(page_width,0.0,scan.capitalize())
			pdf.ln(10)

			reader = csv.reader(f)

			pdf.set_font('Times', '', 12)
			col_width = page_width/3
			th=pdf.font_size
			for row in zip(*reader): 
				pdf.cell(col_width, th, row[0],border=1)
				pdf.cell(col_width, th, row[1],border=1)
				pdf.ln(th)        

			pdf.add_page()


# Finalize pdf
pdf.output(fldrs[0] + '_' + fldrs[1] + '_MRIQC_IQMs.pdf','F')