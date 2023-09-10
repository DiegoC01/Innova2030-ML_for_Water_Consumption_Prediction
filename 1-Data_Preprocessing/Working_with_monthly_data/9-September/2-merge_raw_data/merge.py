import os

path = "1-Data_Preprocessing/Working_with_monthly_data/9-September/1-raw_data/"
file_paths = os.listdir(path)
print(file_paths)

with open('1-Data_Preprocessing/Working_with_monthly_data/9-September/2-merge_raw_data/merged_data/merged_data_to_rename.csv', 'w', encoding='utf-8') as output_file:
    for file_path in file_paths:
        with open(path+"/"+file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                output_file.write(line)
            output_file.write('\n')