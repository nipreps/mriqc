#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pdfgen, os, glob, asyncio
from html.parser import HTMLParser
from pyppeteer import launch

#def patch_websockets():
#	import websockets.client
#	original_method = websockets.client.connect

#	def new_method(*args, **kwargs):
#		kwargs['ping_interval'] = None
#		kwargs['ping_timeout'] = None
#		return original_method(*args, **kwargs)

#    websockets.client.connect = new_method

#patch_websockets()

async def main():
	#browser = await launch(headless=True, args=['--no-sandbox'])
	await pdfgen.from_file("tmp_html.html",x.replace('.html','.pdf'))

html_out = glob.glob('sub-*.html')

for x in html_out:
	with open(x,"r") as file:
		file_data = file.readlines()
	first_line=[]
	last_line=[]
	line_number = 0
	# locate and remove ratings widget
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
	# save html string to temporary file for conversion
	tmp_file = open("tmp_html.html","w")
	tmp_file.write(html)
	tmp_file.close()
	browser = launch(
		executablePath='/usr/bin/google-chrome-stable', 
		headless=True, 
		args=[
			'--no-sandbox',
			'--single-process',
			'--disable-dev-shm-usage',
			'--disable-gpu',
			'--no-zygote'
		])
	#Apply html to pdf conversion
	asyncio.get_event_loop().run_until_complete(main())
	browser.close()
	#delete temporary file
	if os.path.exists("tmp_html.html"):
		os.remove("tmp_html.html")
	else:
		print("The file does not exist.") 

#	with Xvfb() as xvfb:
		#pdfkit.from_string(html, x.replace('.html','.pdf'),options={"encoding":"utf8","image-dpi": 1920, "disable-smart-shrinking": None, "enable-local-file-access":None})


