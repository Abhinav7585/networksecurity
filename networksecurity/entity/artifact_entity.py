from dataclasses import dataclass
# with the help of dataclass, it just acts like a decorator which probably creates a variable for an empty class
# Lets say in my class i dont have any functions, I just need to havve some class variables defined, we can use this..

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str
