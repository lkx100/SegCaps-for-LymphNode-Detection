import os

# print("Enter Annotated Numbers (' ' separated):", end=" ")
# annotated_numbers = list(map(int, input().split()))
# unique_ann = list(set(annotated_numbers))
# regex_pattern = "(" + "|".join([f"1-{num:03d}\\.dcm" for num in unique_ann]) + ")"
# print(regex_pattern)

num = int(input())
# make num a 3 digit number always. Like if num is 3, it should be 003
num = str(num).zfill(3)
txt_file_path = f"mnt/d/manifest-1680277513580/Others/MED_ABD_LYMPH_ANNOTATIONS/ABD_LYMPH_{num}/ABD_LYMPH_{num}_lymphnodes_indices.txt"

# txt_file_path = input("Enter the path to the text file: ")

# if txt_file_path[-1] or txt_file_path[0] == '"':
#     txt_file_path = txt_file_path[1:-1]  # Remove quotes if present

# Check if the file exists
if not os.path.isfile(txt_file_path):
    print(f"File not found: {txt_file_path}")
    exit(1)

# Check if the file is empty
if os.path.getsize(txt_file_path) == 0:
    print(f"File is empty: {txt_file_path}")
    exit(1)

# Check if the file has the correct format
with open(txt_file_path, "r") as file:
    lines = file.readlines()
    if not all(len(line.split()) >= 3 for line in lines):
        print(f"File format is incorrect: {txt_file_path}")
        exit(1)

# Read the file and extract the third column
with open(txt_file_path, "r") as file:
    annotated_numbers = [int(line.split()[2]) for line in file.readlines()]  # Extract third column

# Format numbers with zero-padding and generate regex
regex_pattern = "(" + "|".join([f"1-{num:03d}\\.dcm" for num in annotated_numbers]) + ")"

print(regex_pattern)  # Use this regex in ImageJ


