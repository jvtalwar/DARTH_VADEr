"""Pre-Data-Loader to split data into files"""

import pandas as pd
from typing import Optional
from pyarrow import feather
import logging
import json
import os

logging.getLogger().setLevel(logging.INFO)

class DataWriter:
    """Pre-Data Loader to get away from PyTorch wrapper"""

    def __init__(self, genotypes: pd.DataFrame, nameFormat: Optional[str] = None, idToFilenamePath: Optional[str] = None):
        """Initializes data, counter variable, and filename format """
        self.genotypes: pd.DataFrame = genotypes
        self.fileNameFormat: str = "genotype_data_record_{file_id}.feather" if nameFormat is None else nameFormat
        self.idToFilenamePath: str = idToFilenamePath
        

    def splitData(self) -> None:
        """
        Splits data into individual files by initialized file name

        @param idToFilenamePath - a path in which to store the dict 
        """
        
        writePath = os.path.join(self.idToFilenamePath, "idToFile.json")
        if os.path.exists(writePath):
            logging.warning("ID-File dictionary already exists, utilizing {} individual pre-processed feathers...".format(len([el for el in os.listdir(self.idToFilenamePath) if el.endswith(".feather")])))
            return 
                          
        # Id to filenames dictionary
        idsToFileNames = {}

        # Traversing genotypes, writing fo files, adding record to dict
        for curr_id in self.genotypes.columns:
            # Building file name and adding to dict
            currFileName = self.fileNameFormat.format(file_id = curr_id)
            idsToFileNames[curr_id] = currFileName
            feather.write_feather(df = self.genotypes[curr_id].to_frame(), dest = os.path.join(self.idToFilenamePath, currFileName))

        logging.info("{} feather files written to {}".format(len([el for el in os.listdir(self.idToFilenamePath) if el.endswith(".feather")]), self.idToFilenamePath))
        
        # If filename not provided, warning user
        if self.idToFilenamePath is None:
            # Checking if filename provided, otherwise forming our own in current working directory
            logging.warning(f"No Path provided, storing map file in current directory at {os.getcwd()}/idToFile.json")
            writePath = f"{os.getcwd()}/idToFile.json"           
        
        
        # Writing json to file
        with open(writePath, "w") as writer:
            json.dump(idsToFileNames, writer)


        # Yielding a success message
        logging.info("GREAT SUCCESS!!! (Successfully wrote data to files)")

