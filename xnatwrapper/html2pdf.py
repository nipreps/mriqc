#install pdfkit and wkhtmltopdf 

import pdfkit
from html.parser import HTMLParser

with open('sub-229415_ses-1_T1w.html',"r") as html_out:
    file_data = html_out.readlines()

parsed_data = file_data[:12781]+file_data[12795:]

myString = ''
for i in parsed_data:
    myString += i

pdfkit.from_string(myString, 'html.pdf')