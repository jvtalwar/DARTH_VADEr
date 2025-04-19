'''
About: Python script to compile chromosome-partitioned raw files into a comosite-level raw (i.e., non-zscored) feather file.
'''

import pandas as pd
from itertools import islice
import argparse
import os
import glob
import tqdm

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def main(args):
    file_path=args.directory
    
    #iterate over chromosomes
    files=sorted(glob.glob(os.path.join(file_path, (args.prefix + "*.raw"))))
    print("Compiling the following {} files:".format(len(files)))
    
    compiled_raw = pd.read_csv(files[0],delim_whitespace=True)
    compiled_raw = compiled_raw.drop(["FID","PAT","MAT","SEX","PHENOTYPE"], axis=1)
    print(files[0], compiled_raw.shape)

    for x in tqdm.tqdm(files[1:]):
        
        raw = pd.read_csv(x,delim_whitespace=True)
        print(x, raw.shape)
        
        if ("X" in x) and ("ukbb" in x.lower()):
            print("using mapping for UKBB X file")
            mappingx = pd.read_csv("../../Data/mapping_Files/UKBiobank_linker.txt", delim_whitespace = True) #Path to mapping file for UKBB X chromosome which employs different individual IDs
            mapx = dict(zip(mappingx["FID_Salem"], mappingx["FID"]))
            raw["FID"] = raw["FID"].map(mapx)
            raw["IID"] = raw["IID"].map(mapx)
            raw = raw[~raw["FID"].isnull()]
            raw = raw[~raw["IID"].isnull()]
        
        raw = raw.drop(["FID","PAT","MAT","SEX","PHENOTYPE"], axis=1)
        compiled_raw = pd.merge(compiled_raw, raw, on = ["IID"], how = "left")
        print(f"Completed loading file: {x}")
    
    compiled_raw = compiled_raw.set_index("IID")
    compiled_raw = compiled_raw.T
    compiled_raw = compiled_raw.reset_index()
    compiled_raw.columns = [str(x) for x in compiled_raw.columns]
    compiled_raw.to_feather(args.out_file) #export compiled raw file
    print("Outputting {} X {} table to {}".format(compiled_raw.shape[0], compiled_raw.shape[1], args.out_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='', help='if there is a prefix to also look for in the raw files')
    parser.add_argument('--directory', type=str, help='where raw files are located')
    parser.add_argument('--out_file', type=str, help='output file path')
    args = parser.parse_args()
    main(args) 
