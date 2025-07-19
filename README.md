### Network Security Projects for Phishing data


Main aim of this Machine Learning project is to create a model wherein whenever I get this specific dataset(i.e from features in the dataset) from a specific website, I need to predict whether the website is "fake" or whether it is "Malicious website" or not

This is our problem statement

Step-1: ETL pipeline
    We have the datset in our local machine, now we will create a ETL pipeline wherein we will read that dataset,do some transformation and finally convert it to a JSON, then we will insert the entire data into my MongoDB database(MongoDB atlas)
    Specifically MongoDB will be in cloud and Atlas actually provides some free account to create your MongoDB database and use them for some number of requests
    
Step-2: Data Ingestion

    First we require "Data Ingestion Config"(contains all the info like Data Ingestion Dir, Feature Store File path, Training File path, Testing File path, Train Test split ratio, Collection Name)

    Next we will create another file, where be reading from MongoDb 

    Once we read all the data, we take all the "data ingestion config" info into "Data Ingestion Artifact"

    Data Ingestion Artifact => Is the output of Data Ingestion Config
     
    MongoDB --> Raw data(Stored in Data Ingestion Artifacts) --> Feature Engineering(drop unncessary columns etc) --> Split into train and test --> Data Ingestion Artifact
    

