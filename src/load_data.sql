CREATE DATABASE ai_planning_project;
USE ai_planning_project;

CREATE TABLE attractions (
    aid INT NOT NULL,
    aname VARCHAR(100) NOT NULL,
    expenditure INT,
    timespent FLOAT,
    CONSTRAINT attractions_pk PRIMARY KEY (aid),
    CONSTRAINT unique_name UNIQUE (aname)
);


CREATE TABLE tourists (
    tid INT NOT NULL,
    type VARCHAR(100) NOT NULL,
    CONSTRAINT tourist_pk PRIMARY KEY (tid),
    CONSTRAINT unique_type UNIQUE (type)
);


CREATE TABLE preference (
    type VARCHAR(100) NOT NULL,
    aname VARCHAR(100) NOT NULL, 
    scale INT,
    CONSTRAINT preference_pk PRIMARY KEY (type, aname),
    CONSTRAINT preference_fk1 FOREIGN KEY (type) REFERENCES tourists(type),
    CONSTRAINT preference_fk2 FOREIGN KEY (aname) REFERENCES attractions(aname)
);


CREATE TABLE foodcentre (
    fid INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    expenditure INT,
    timespent FLOAT,
    rating FLOAT,
    type VARCHAR(100),
    bestfor VARCHAR(100),
    highlights VARCHAR(1000),
    address VARCHAR(1000),
    CONSTRAINT foodcentre_pk PRIMARY KEY (fid)
);
# check the path to local postition
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/PROJECT/attractions.csv' 
INTO TABLE attractions 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
SELECT * FROM attractions;


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/PROJECT/tourists.csv' 
INTO TABLE tourists 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
SELECT * FROM tourists;


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/PROJECT/foodcentre.csv'  
INTO TABLE foodcentre
FIELDS TERMINATED BY ','  
ENCLOSED BY '"'  
LINES TERMINATED BY '\r\n'  
IGNORE 1 ROWS;
SELECT * FROM foodcentre;


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/PROJECT/preference.csv' 
INTO TABLE preference
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
SELECT * FROM preference;
