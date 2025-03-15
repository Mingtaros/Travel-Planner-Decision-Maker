# Singapore Travel Itinerary Project - Database Setup

This repository contains the Docker setup for the AI Planning Project database.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Directory Structure

```
travel-planner-decision-maker/
├── docker-compose.yml
├── init/
│   └── init.sql
├── data/
│   ├── attractions.csv
│   ├── tourists.csv
│   ├── foodcentre.csv
│   └── preference.csv
└── mysql-data/  (will be created automatically)
```

## Setup Instructions

1. Clone this repository
2. Place your CSV files in the `data` directory
3. Start the database container:

```bash
docker-compose up -d
```

4. Verify the database is running:

```bash
docker-compose ps
```

5. Connect to the database:

```bash
docker exec -it ai_planning_project_db mysql -u planner -pplannerpassword
```

6. Run the python script:

```bash
python ./src/load_data.py
```

7. Verify data was loaded:

```sql
USE ai_planning_project;
SELECT * FROM attractions LIMIT 5;
SELECT * FROM foodcentre LIMIT 5;
```

## Stopping the Database

```bash
docker-compose down
```

To completely remove volumes (will delete all data):
```bash
docker-compose down -v
```

## Connecting from Your Application

Use the following parameters to connect:

- Host: localhost (or your Docker host IP)
- Port: 3306
- Database: ai_planning_project
- Username: planner
- Password: plannerpassword