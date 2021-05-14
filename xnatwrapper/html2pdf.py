#!/usr/bin/env python3
#install pdfkit and wkhtmltopdf 

import pdfkit, os, glob
from html.parser import HTMLParser

html_out = glob.glob('*.html')

for x in html_out:
	with open(x,"r") as file:
		file_data = file.readlines()

	first_line=[]
	last_line=[]
	line_number = 0

	for line in file_data:
		line_number += 1

		if '<div id="rating-menu" ' in line:
			first_line.append((line_number-1))

		if '<label class="btn btn-outline-success">' in line:
			last_line.append((line_number+1))

	#add in first and last line variables
	parsed_data = file_data[:first_line[0]]+file_data[last_line[0]:]

	myString = ''
	for i in parsed_data:
		myString += i

	pdfkit.from_string(myString, x.replace('.html','.pdf'))