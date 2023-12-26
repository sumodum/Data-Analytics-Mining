## Setting up 
  
  Ensure python version is at least 3.8! 
  
  MAC OS: 
  1. Download the repository and locate it in terminal. 
  2. Install virtualenv using ```sudo pip install virtualenv```
  3. Create virtual environment using ```virtualenv env``` then start it using ```source env/bin/activate```
  4. Download required libraries and packages using ```pip install -r requirements.txt```
  
  WINDOWS:
  1. Download the repository and locate it in terminal.
  2. Install virtualenv using ```pip install venv```
  3. Create virtual environment using ```venv env``` then start it using ```.\\env\Scripts\activate```
  4. Download required libraries and packages using ```pip install -r requirements.txt```
  
## Running CBA classifier
  Ensure DATAPATH and MINSUP is updated in config.py
  
  1. Enter ```python3 cba.py``` to run our python file. Make sure updating of data path is done as shown below!
  2. Alternatively, ```python3 cba_bagging.py``` to run our code where we implement bagging. 
   
   Replace lines in config.py with the below for each dataset:

### Breast
```
DATAPATH = 'csv_datasets/Breast Cancer/breast_cba.csv'
MINSUP = 0.2

```

### Mushroom
```
DATAPATH = 'csv_datasets/Mushroom/mushroom_cba.csv'
MINSUP = 0.5

```

### Tic Tac Toe
```
DATAPATH = 'csv_datasets/TicTacToe/tictactoe_cba.csv'
MINSUP = 0.2

```

### Ionosphere
```
DATAPATH = 'csv_datasets/Ionosphere/ionosphere_cba.csv'
MINSUP = 0.2

```

### Banknotes
```
DATAPATH = 'csv_datasets/Bank Note Auth/data_banknote_authentication_cba.csv'
MINSUP = 0.2

```

### CRX 
```
DATAPATH = 'csv_datasets/Japanese Credit Screening/crx_cba.csv'
MINSUP = 0.2

```
### Mammographic
```
DATAPATH = 'csv_datasets/Mammographic/mammographic_masses_cba.csv'
MINSUP = 0.2

```

### Hepatitis
```
DATAPATH = 'csv_datasets/Hepatitis/hepatitis_cba.csv'
MINSUP = 0.2

```

## Running other classifiers
  
  1. Update DATAPATH in config.py with the csv file used by the other classifiers ending with '_others':
   For example, for Breast Cancer dataset,
   ```
   DATAPATH = 'csv_datasets/Breast Cancer/breast_others.csv'
   ```
  2. Enter ```python3 decision_tree.py``` to run decision tree classifier.
  3. Enter ```python3 random_forest.py``` to run random forest classifier.
  4. Enter ```python3 svm.py``` to run support vector machine classifier.