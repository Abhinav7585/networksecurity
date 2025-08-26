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


Step-3: Data Tranformation
    Here we implement DataTransformationConfig, DataTransformationComponent and DataTransformationArtifact is its output..

    In Data Transimission, w.k.t firstly we create out "Data Transformation Config". In Data Transimission Config we will be having the info about:
        - Data Validation Dir
        - Valid Data Dir
        - Invalid Data Dir
        - Valid Train file path
        - Invalid Train file path
        - Drift report file path
    We will be having this info ,this info already came from data validation

    Then we go to next ste where we start our implementation of data Transformation i.e "Initiate Data Transformation"
    As soon as we initiate it we go to the data validation artifact and then we "Read all the train and test data"

    Then we do Data Transformation step..

    In Data Transformation, what we basically do is that we will be take our "train dataframe" and we will drop the target column => Then we will get "Target Feature Train Dataframe" and "Input Feature Train Dataframe" and then we can combine these and create train array

    Then we can use "SMOTE tomek" if our dataset is unbalanced ..But our dataset is balanced..So here no need to use..
    'SMOTE tomek' is a Feature Engineering process

    But we will take the "training Dataset" and we can see some "NAN" values. So we will try to replace this Nan value with something
    For that we are going to implement or use one "imputer technique".
    It can be 
        - Robust Scaler technique
        - Simple Imputer technique
        - KNN imputer
    we are going to use KNN imputer. We are going to create a pipeline.
    Once we do this we will get "Preprocessor Object" => "preprocessing.pkl"
    We will alos apply this particular step to my test data also..
    This pkl file also be saved and will create a folder called as "Data Tranformation", Inside my "Artifacts" folder....

    Basically here we are going to handle Missing values and we will try to create a pipline for KNN imputer

    For train data we apply "fit_transform" and for test data we apply "transform"

    
Step-4: Model Trainer Component
    Now we have "data tranformation artifacts"
    Data Transformation basically means if we want to do any "Feature Engineering",In our case, we have applied how to replace "Nan values". We used 
    "KNNImputer" to replace Nan Values
    We Created the pickle file, so that this pickle file can be used anywhere, based on our model training
    Now after we got the data transformation artifact, now we really need to focus on the model trainer component..
    Now again w.r.t our model trainer component ,we should create our "modelTrainerConfig"

    This "modelTrainerComponent" the "dataTransformationArtifact" should be provided as input and then we need to alos give input w.r.t to the "modelTrainerConfig".

    This "modelTrainerConfig" should have the details after training the model, where we should probably save thhe model itself...In which folder and all..

    Then after the "modelTrainerComponent" is done, the output of "modelTrainerComponent" is "modelTrainerArtifact"
    Inside this artifact, we specifically focus on 2 important things i.e
        - We create our "model.pkl" file =>Once our model is trained,we get this..
        - And from "dataTranformationArtifact", whichever pickle file we have used for our pipeline. So this pickle file should also be coming over here
    
    Architecture of "ModelTrainerComponent"

    So initially when we start with the "modelTrainerConfig" .These are inital details:
        - model trainer dir => Location where we will be saving our model
        - trained model file path => Entire model file path where we really wnat to save 
        - expected accuracy 
        - model config file path => Info related to model config can be stored here...
    These are the basic info that I will be giving when we initiate the "modelTrainerConfig"
    Then we "initiate the Model Training"

    Then we "Load numpy array data" and that we are basically taking to take it from "dataTranformationArtifact"

    Then we do a split for train and test array i.e "x train array","y train array","x test array","x test array"

    Then we will train our model w.r.t our training data and then we will try multiple models. Once we get the best model and we will try to compare the best score.
    So if I probably find out the best model w.r.t best score, we are going to probably take that and we will convert that into a pickle file...
    To find best model we use "calculate metrics"



NOTE:
    I developed this project based on industry standards(acc to what I researched)
    -> Now we will be able to track all these things in our "MLFlow"
    -> "MLFlow" is an opensorce tool to completely handle your ML lifecycle of a ML project w.r.t experimentation, w.r.t selecting multiple models, comparing multiple models, comparing multiple metrics and all


MLFlow =>
    -> Once we got the "best model", whatever "classfication_metric" we usually get, we really need to make sure that we track the entire thibg in MLflow..

    -> "Mlflow" is an opensource tool which is very useful to manage the entire lifecycle of a data science project..
    Here you'll be able to make sure that you put all your performance metrics you can, any no.of experiments you are probably doing, finding the best model along with any classification metrics that is there..
    You can actually store in all this kind of experiments with help of MLflow
    And you can also visulize them and compare them w.r.t various experiments itself.

    -> "mlruns" folder will be created, inside that we will be able to see all the experiments that we performed and its details

    -> Inorder to check this particular experiment, How do you check it with the help of MLflow ??
    
    -> In command prompt type "mlflow ui" => we will get an url opened and if we click on it, There we will be able to see the entire experiment logged in this URL..

    -> With MLflow you will be able to see the entire logs of experiments, what performance metrics it captured and which is your model file

    -> If we change our algorithm and run it again, then one more file will be created 

    -> We can also compare different experiments..


Whenever we use MLflow tracking we used to probably track in our local folder,
But how can we track all the info related to MLflow, where we are logging about experiments,logging about the best model in a "remote repository".
-> "Remote repository" means it will not be in local, but it will be in a specific URL..,where you can see it out.
-> For that we use "Dagshub".
-> Dagshub just gives you remote repositories where you can run millions of experiments as you like.
-> Similar ot github but here you can track many things..

-> AIM for Dagshub: Whatever "mlruns" that we probably run in my local, it should be available in dagshub..
-> SO whenever I run my code in local , my experiment should be logged in dagshub.. 

-> Whenever I probably go and run this code, "mlflow.log_metric()" and all knows where we really need to go ahead and commit, where we need to create our "mlruns"
or which remote repo we need to track our entire data.

-> This time my "mlruns" will not be created in the local machine 

-> When we run, it first accesses our repo in daghub and it has initialized MLflow to track this experiment

-> Entire training will happen in the local machine but once the entire MLflow tracking code will get executed, that entire data will get stored over here(in dagshub) i.e the "mlruns" folder will be stored there in dagshub.

-> Why are we doing this?
    -> MLflow is completely opensource and I have a remote repository which is also opensource, which we can use it for free and we can directly give that URL to anyone to probably track the experiment.
    
    -> So if you take a paid account of Dagshub, then lets say you want to probably share the entire reports of the performance metrics or anything that you want.
    
    ->With the help of this remote repo, we will be able to just give the URL. People will be able to check it out.

    -> This is how collaboratively you will be able to work in a team.

    -> Therefore our main idea is:
    Whatever commits or whatever tracking of experiments that we are doing, we are not doing it in local, Even though MLflow runs in local, we are instead doing it in a remote repository which can probably be shared with anyone 














