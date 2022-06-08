import csv


## Input 
RPM = 600
Input_File_name = 'knoy_mpu_3_train_' + str(RPM) 
Output_File_Name = 'knoy_mpu_3_'+ str(RPM)

z = open(Input_File_name+'.csv','w')
z.writelines("Time"+','+"X"+','+"Z"+','+"defect"+'\n')
with open(Output_File_Name +'.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
       # print(row) 	
        z.writelines('"' +str(6) +'-'+ str(30) + '"' + ',' + row['x'] + ',' + row['z'] + ',' + str(3) + '\n')
        line_count += 1
    print(f'Processed {line_count} lines.')
