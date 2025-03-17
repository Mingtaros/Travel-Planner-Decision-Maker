-- MySQL dump 10.13  Distrib 9.2.0, for Linux (x86_64)
--
-- Host: localhost    Database: ai_planning_project
-- ------------------------------------------------------
-- Server version	9.2.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `waypoints`
--

DROP TABLE IF EXISTS `waypoints`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `waypoints` (
  `id` int NOT NULL AUTO_INCREMENT,
  `location_id` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `name` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `type` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL,
  `latitude` double NOT NULL,
  `longitude` double NOT NULL,
  `address` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_location` (`location_id`)
) ENGINE=InnoDB AUTO_INCREMENT=82 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `waypoints`
--

LOCK TABLES `waypoints` WRITE;
/*!40000 ALTER TABLE `waypoints` DISABLE KEYS */;
INSERT INTO `waypoints` VALUES (54,'A4','ArtScience Museum','attraction',1.2862738,103.8592663,'6 Bayfront Ave, Singapore 018974','2025-03-17 08:11:18','2025-03-17 08:11:18'),(55,'A6','Asian Civilisations Museum','attraction',1.2874969,103.8513861,'1 Empress Pl, Singapore 179555','2025-03-17 08:11:18','2025-03-17 08:11:18'),(56,'A14','Chinatown','attraction',1.2814942,103.8448202,'Chinatown, Singapore','2025-03-17 08:11:18','2025-03-17 08:11:18'),(57,'A11','Clarke Quay','attraction',1.2906024,103.8464742,'3 River Valley Rd, Singapore 179024','2025-03-17 08:11:18','2025-03-17 08:11:18'),(58,'A13','Esplanade','attraction',1.2897934,103.8558166,'1 Esplanade Dr, Singapore 038981','2025-03-17 08:11:18','2025-03-17 08:11:18'),(59,'A9','Fort Canning Park','attraction',1.2943876,103.8458033,'Singapore','2025-03-17 08:11:18','2025-03-17 08:11:18'),(60,'A1','Gardens by the Bay','attraction',1.2815683,103.8636132,'18 Marina Gardens Dr, Singapore 018953','2025-03-17 08:11:18','2025-03-17 08:11:18'),(61,'A15','Little India','attraction',1.3065597,103.851819,'Little India, Singapore','2025-03-17 08:11:18','2025-03-17 08:11:18'),(62,'A2','Marina Bay Sands SkyPark Observation Deck','attraction',1.2852044,103.8610313,'10 Bayfront Ave, Singapore 018956','2025-03-17 08:11:18','2025-03-17 08:11:18'),(63,'A10','Merlion Park','attraction',1.2867449,103.8543872,'1 Fullerton Rd, Singapore 049213','2025-03-17 08:11:18','2025-03-17 08:11:18'),(64,'A5','National Gallery Singapore','attraction',1.2902217,103.8515167,'Singapore 178957','2025-03-17 08:11:18','2025-03-17 08:11:18'),(65,'A7','National Museum of Singapore','attraction',1.296613,103.8485091,'93 Stamford Rd, Singapore 178897','2025-03-17 08:11:18','2025-03-17 08:11:18'),(66,'A12','Orchard Road','attraction',1.3048205,103.8321984,'Orchard Rd, Singapore','2025-03-17 08:11:18','2025-03-17 08:11:18'),(67,'A8','Peranakan Museum','attraction',1.2943669,103.8490391,'39 Armenian St, Singapore 179941','2025-03-17 08:11:18','2025-03-17 08:11:18'),(68,'A16','Singapore Botanic Gardens','attraction',1.3138397,103.8159136,'1 Cluny Rd, Singapore 259569','2025-03-17 08:11:18','2025-03-17 08:11:18'),(69,'A3','Singapore Flyer','attraction',1.2892988,103.8631368,'30 Raffles Ave., Singapore 039803','2025-03-17 08:11:18','2025-03-17 08:11:18'),(70,'H1','Lau Pa Sat','hawker',1.2806753,103.8503722,'18 Raffles Quay, Singapore 048582','2025-03-17 08:11:18','2025-03-17 08:11:18'),(71,'H2','Amoy Street Food Centre','hawker',1.2794385,103.8466789,'7 Maxwell Rd, Singapore 069111','2025-03-17 08:11:18','2025-03-17 08:11:18'),(72,'H3','Hill Street Tai Hwa Pork Noodle','hawker',1.3051265,103.8625357,'466 Crawford Ln, #01-12 Block 466, Singapore 190466','2025-03-17 08:11:18','2025-03-17 08:11:18'),(73,'H4','Chinatown Complex Food Centre','hawker',1.2825863,103.8430923,'335 Smith St, #02-126, Singapore 050335','2025-03-17 08:11:18','2025-03-17 08:11:18'),(74,'H5','Odette','hawker',1.2902217,103.8515167,'Singapore 178957','2025-03-17 08:11:18','2025-03-17 08:11:18'),(75,'H6','JAAN by Kirk Westaway','hawker',1.2934948,103.8534342,'2 Stamford Rd, Lvl 70 #70-01, Singapore 178882','2025-03-17 08:11:18','2025-03-17 08:11:18'),(76,'H7','Cut by Wolfgang Puck','hawker',1.2851095,103.8595667,'10 Bayfront Ave, #02-03 The Shoppes, Marina Bay Sands, Singapore 018956','2025-03-17 08:11:18','2025-03-17 08:11:18'),(77,'H8','Waku Ghin','hawker',1.2837575,103.8591065,'10 Bayfront Ave, Singapore 018956','2025-03-17 08:11:18','2025-03-17 08:11:18'),(78,'H9','Sungei Road Laksa','hawker',1.3066822,103.8578231,'27 Jalan Berseh, #01-100, Singapore 200027','2025-03-17 08:11:18','2025-03-17 08:11:18'),(79,'H10','Muthu\'s Curry - Little India','hawker',1.3098372,103.8521672,'138 Race Course Rd, Singapore 218591','2025-03-17 08:11:18','2025-03-17 08:11:18'),(80,'H11','Tekka Centre','hawker',1.3063839,103.8507014,'665 Buffalo Rd, Singapore 210665','2025-03-17 08:11:18','2025-03-17 08:11:18'),(81,'H12','Maxwell Food Centre','hawker',1.2802419,103.8449058,'1 Kadayanallur St, Singapore 069184','2025-03-17 08:11:18','2025-03-17 08:11:18');
/*!40000 ALTER TABLE `waypoints` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-03-17  8:25:41
