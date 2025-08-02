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
    
Step-3: Data Validation
   The only input to data validation component is "data ingestion artifact"

   And in Data Validation component what we actually do is we create a "data validation config" so we will have necessary path information

   And then, after we complete this data validation component ,we get "Data validation artifact" as the output.
   
   Q. What we basically do in Data Validation?
   ANS: Here whenever we read data from MongoDB, the most important thing is that my "DATA SCHEMA should not change" 

   "Schema" basically means, Suppose I have 10 features in my dataset, This 10 features should always be there. If these 10 features are not there (or) One of the feature is of any other datatype, "Then we will not be able to train our model properly"

   Along with that we also need to check for "CHANGE IN DATA DISTRIBUTION" i.e "Data Drift"
   If there is Data Drift we cannot use that same data properly to train our model because there is huge difference between old data and new data (THIS IS A BIG ISSUE.)

   What should we do for avoiding Data Drift?
   We should create soething called as "DATA DRIFT REPORT".
   This report tells developers that whatever data that is coming , the distribution is completely getting changed.

   So this all checks are done in DATA VALIDATION
        i. Same Schema => Same no. of Features and same datatypes
        ii. Data Drift => Check whether the distribution of new data is same or not,
        when we compare it with the training data(to train our model).
        iii. Validate no. of columns, Numerical columns exist or not etc..
    
    What all folders do we require?
    We require:
        -> Data Validation Directory
        -> Valid Data Directory
        -> Invalid Data Directory
        -> Valid Train file path
        -> Invalid Train file path (i.e if my train file that we are getting, doesnt have same no. of columns, Then we put it in "Invalid Train file path")
        -> Invalid Test file path
        -> Drift Report file path
    So initially inside "DATA VALIDATION CONFIG" WE DEFINE ALL THIS PARTICULAR PATHS

    Next we will "INITIATE DATA VALIDATION". To initiate data validation, from "Data Ingestion Artifact" from the "Ingested" folder, we are going to read "train.csv and test.csv".
    And w.k.t "Data Ingestion Artifact" is given as input to Data Validation.

    Next step here we have is "Validate No. of columns" function (First we check Train data whether we have same no. of columns or not and then for test data..) and this function returns status like true or false

    Next one more check we do after this is "Numerical column exist or not" both for train data and test data and return some kind of status..

    Then next we go to "Validation Status" stage (combination of all status) => If False, we get "Validation Error" and If True , we go to next step ie "Detect Data Drift" to detect if there is change in distribution or not. There is a MATHEMATICAL WAY to check change in ditribution

    So we will compare previous dataset with which we trained our model and new dataset that we are going to get.
    From this we get "Drift Status". If False, then distribution is almost similar. So from here we are going to generate "report.YAML" file along with this we are going to return our 
    Validation status, 
    Valid Train file path, 
    Invalid Train file path, 
    Invalid Test file path, 
    Drift Report file path
    And this report.YAML will be shared inside articat folder where we will be creating a folder "Data validation Artifact"
















