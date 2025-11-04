import pandas as pd
from shutil import copyfile
from pathlib import Path
import re
from io import StringIO

# Paths
BASE_FOLDER = "computer_vision_diagnostic"
CSV_TRAIN = f"{BASE_FOLDER}/data/raw/csv/mass_case_description_train_set.csv"
CSV_TEST = f"{BASE_FOLDER}/data/raw/csv/mass_case_description_test_set.csv"
IMG_DIR = Path(f"{BASE_FOLDER}/data/raw/jpeg")
TRAIN_DIR = Path(f"{BASE_FOLDER}/data/train")
TEST_DIR = Path(f"{BASE_FOLDER}/data/test")
UPDATED_DIR = Path(f"{BASE_FOLDER}/data/updated")
CSV_UPDATED_TRAIN = f"{UPDATED_DIR}/mass_case_description_train_set.csv"
CSV_UPDATED_TEST = f"{UPDATED_DIR}/mass_case_description_test_set.csv"

# 1. cleanning the file to remove double quotes, empty new lines and replace spaces to underscore on header
def clean_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Removing double quotes in all new lines
    content = re.sub(r'"\s*\n', '\n', content)
    content = re.sub(r'\n"', '\n', content)
    content = content.replace('""', '"')

    # 2. Removing empty lines
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    cleaned_content = '\n'.join(lines) + '\n'

    # 3. Read file with pandas
    df = pd.read_csv(StringIO(cleaned_content), skipinitialspace=True)

    # 4. Replace space by underscore in headers
    df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

    # 5. Save formatted file
    df.to_csv(output_path, index=False)
    print(f"CSV limpo salvo: {output_path}")

# 2. Call function above to clean-up both files (test and train)
clean_csv(CSV_TRAIN, CSV_UPDATED_TRAIN)
clean_csv(CSV_TEST, CSV_UPDATED_TEST)

# 3. Create destination folder
for dir_path in [TRAIN_DIR/"BENIGN", TRAIN_DIR/"MALIGNANT", TEST_DIR/"BENIGN", TEST_DIR/"MALIGNANT"]:
    dir_path.mkdir(parents=True, exist_ok=True)


# 4. Organize exam files in folders
def organize(csv_file, dest_dir):
    df = pd.read_csv(csv_file)
    print(f"Processing {len(df)} entries from {csv_file}...")

    for _, row in df.iterrows():
        # 1. Get file path
        raw_path = row['image_file_path']  # This path contains more information than just the file path"

        # 2. Spliting the content by slash
        parts = raw_path.strip().split('/')
        if len(parts) < 3:
            print(f"Bypassing invalid line: {raw_path}")
            continue

        # 3. The third path of the content has the folder name
        jpg_folder_name = parts[2]  # ex: "1.3.6.1.4.1.9590.100.1.2.245063149211255120613007755642780114172"
        
        # 4. Building complete folder path
        jpg_folder = IMG_DIR / jpg_folder_name

        # 5. Searching .jpg files in the folder
        jpg_files = list(jpg_folder.glob("*.jpg"))
        if not jpg_files:
            print(f"No .jpg file found in: {jpg_folder}")
            continue

        # 6. Getting first .jpg file
        img_path = jpg_files[0]
        jpg_name = img_path.name

        # 7. Defining label
        label = "MALIGNANT" if row['pathology'] in ['MALIGNANT', 'MALIGNANT_B'] else "BENIGN"
        dest = dest_dir / label / jpg_name

        # 8. Copying the file
        try:
            copyfile(img_path, dest)
        except Exception as e:
            print(f"Error to copy {img_path}: {e}")

# 5. Executing the code
print("Organizing training...")
organize(CSV_UPDATED_TRAIN, TRAIN_DIR)
print("Organizing testing...")
organize(CSV_UPDATED_TEST, TEST_DIR)
print("ORGANIZATION COMPLETED WITH SUCCESS!")