
import json,csv


#Import json file
with open('/home/rdlawless/Documents/MRIQC/outputs/sub-229415/ses-1/anat/sub-229415_ses-1_T1w.json','r') as file:
	json_data=json.load(file);

json_data.pop('bids_meta','provenance');

csv_file = open('csv_file.csv','w');

write=csv.writer(csv_file);

write.writecol(json_data.keys());
write.writecol(json_data.values());

csv_file.close();
