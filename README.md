# 2021-BA-Franka-Knoch Overview

This repository refers to the bachelor's thesis: \
\"Comparison of Group-based Outlier Detection Techniques for Large-scale Environmental Sensor Networks\". \
It includes the code for obtaining insights into the Outlier Detection Techniques and other stastical analysis and plots used in this thesis.\
The main goal of this project is to analyse the particulate matters dataset measured by an environmental sensor network located in Stuttgart.\
Therefore, the sensors are grouped into smaller clusters via an agglomerative hierarchical clustering algorithm.\
On these clusters the Outlier Detection Techniques Box Plot, Kernel Density Estimation, and Isolation Forest are applied.\
The results, plots, and further statistics are provided by this project.

# Instructions

Follow the insctruction for the database setup, as well as the explanations about the project structure to obtain results.

## Database Setup

1. Install PostgreSQL / MySql (and a GUI like pgadmin, DBeaver, ...)
2. Create a new database (on localhost). Either via the GUI or via the CMD
3. Create a new user with password
4. Assign privileges to the user
5. Download database dump: https://gigamove.rz.rwth-aachen.de/d/id/Mie77HNzt2fCHF Password: ldistuttgart
6. Extract database dump from .zip
7. Link is only valid for a maximum of 14 days(Valid until: 26.05.2021). If link is invalid mail: franka-maria.knoch@stud.uni-bamberg.de
8. Import the database dump

## Sourcecode Setup

1. Pull the repostitory
2. Download / Open preferred IDE (PyCharm, Spyder, ...)
3. Open the project
4. Install requirements for the project from requirements.txt `pip install -r requirements.txt` \
   (Create virtual environment, when versions are conflicting with your current versions.)
5. Create a .env file in the project-folder (same level as \_\_init\_\_.py file containing all methods)
6. Fill the .env file with your previously created database, user, and password. (Copy and fill statements below) \

```bash
user="" #enter user
password=""
host=""
port=""
database=""
```

## Code Structure

The code for obtaining the plots, tables and for executing the Outlier Detection Techniques is stored in python-packages and .py files in the project folder.\
The \_\_init\_\_.py file of the project-folder serves as a starting point and provides further guidelines and explanations.\
The file is separated into subsections the subsections:

- Data Preprocessing
- Data Clustering
- General Statistics
- Box Plot
- Kernel Density Estimation
- Isolation Forest
- Additional Calculations

### Code Execution

1. Load inputs at the top of the \_\_init\_\_.py file (Mark and execute only the inputs, not the whole script!(PyCharm: strg + alt + e.))
2. Select the method you want to execute
3. Read information and comments above the method
4. Mark method and execute the single statement (not the whole script! (Pycharm: strg + alt + e.))
5. Navigate to the script containing the method for further details
   1. Mark whole script and execute whole script to load methods
   2. Go to the starting method & execute it completly or line by line to see the changes in the variables
6. Investigated print-Outputs and Plots, or the variable inspector for results
