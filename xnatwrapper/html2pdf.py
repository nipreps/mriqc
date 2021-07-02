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

	html = ''
	for i in parsed_data:
		html += i

	#xvfb-run --server-args="-screen 0 1024x768x24" /usr/bin/wkhtmltopdf_bin -q $*
	#with Xvfb() as xvfb:

	
	vdisplay = Xvfb(width=1920, height=1080, colordepth=24)
	vdisplay.start()

	pdfkit.from_string(html, x.replace('.html','.pdf'),options={"encoding":"utf8","image-dpi": 1920, "disable-smart-shrinking": None})

	vdisplay.stop()
