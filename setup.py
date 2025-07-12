'''
The setup.py file is an essential part of packaging and
distributing python projects. It is used by setup tools
(or disutils in older python versions) to define the configuration
of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List 
# "find_packages" scans through all the folders available and wherever there is an "__init__.py" file, it is going to consider that particular folder as a package
# "setup" is responsible to provide all info regarding the project

# What we need to do is that whenever we are creating the entire Python package or project package, it is necessary to install all the requirements 
def get_requirements()->List[str]:
    '''

    This function will return list of requirements

    '''
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            # Read lines from the file
            lines=file.readlines()
            # Process each line
            for line in lines:
                requirement=line.strip()
                # Ignore empty lines and -e .
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt")
    
    return requirement_lst
print(get_requirements())
# In this function, what we basically do is that we are reading the requirements.txt file and out of this we are not taking "-e ."

# Our main aim is not just to print requirements.txt but to set up our "metadata"
setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Abhinav Reddy",
    author_email="abhinavreddy1804@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()

)
'''
    When we use "pip install -r requirements.txt" =>it installs all libraries 
    but when it comes to "-e ." what it does is..."-e ." refers to "setup.py" file 
    and when it goes to setup.py file,it is going to execute this entire code where
    it is going to set up this entire Python project as a package.But it is also going
    to do one more thing in this requirement_lst, it is not going to consider "-e ." i.e "-e ."
    is referring to the "setup.file"
 ''' 

'''
    So in short what is basically happening when we write "pip install -r requirements.txt" 
    and it sees "-e ." is present so now lets go and build our package from setup.py. Now in
    setup.py when we are executing this entire code, we will be getting all the packages and
    requirements that is required from requirements.txt

 '''

 '''
    It is not necessary that you need have "-e ." all the time
    We can add "-e ." at the last

    When your complete project is ready and you want to create a final package at that time,
    you can add this.

    This is usually used when you are trying to deply the packages into some cloud or into PyPI as
    a PyPI package all this info is required

 '''