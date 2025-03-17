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

CREATE TABLE transit_fare (
    id INT NOT NULL,
    lower_distance FLOAT,
    upper_distance FLOAT,
    basic_fare FLOAT,
    express_fare FLOAT,
    CONSTRAINT transit_fare_pk PRIMARY KEY (id)
);

-- Setting global variables to allow file loading
SET GLOBAL local_infile = 1;

-- Note: You'll need to load the data manually using the commands below
-- after the container is up and running