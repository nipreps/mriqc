#!/usr/bin/env python
# -*- coding: utf-8 -*-

# install pdfkit and wkhtmltopdf 

import pdfkit, os, glob
from html.parser import HTMLParser
from xvfbwrapper import Xvfb

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

	html_name = ''
	for i in parsed_data:
		html_name += i

	#xvfb-run --server-args="-screen 0 1024x768x24" /usr/bin/wkhtmltopdf_bin -q $*
	#with Xvfb() as xvfb:
	
	vdisplay = Xvfb(width=1280, height=740, colordepth=16)
	vdisplay.start()

	pdfkit.from_string(html_name, x.replace('.html','.pdf'),options={"encoding":"utf8"})

	vdisplay.stop()
