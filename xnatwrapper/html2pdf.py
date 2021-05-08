#!/usr/bin/env python3
#install pdfkit and wkhtmltopdf 

import pdfkit, os, glob
from html.parser import HTMLParser

subject = os.environ["SUBJ"]
ses = os.environ["SES"]
outdir = os.environ["outdir"]

html_out = glob.glob('/*.html')

for x in html_out:
	with open(x,"r") as file:
		file_data = file.readlines()

	parsed_data = file_data[:12781]+file_data[12795:]

	myString = ''
	for i in parsed_data:
		myString += i

	pdfkit.from_string(myString, x.replace('.html','.pdf'))