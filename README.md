# RawChickenCarcasses
## Raw Chicken Carcasses Dataset

## Overview
This dataset contains raw chicken carcass sampling data from various poultry establishments across the United States. The data are analyzed for the presence of Salmonella and Campylobacter, critical for monitoring food safety standards. Additionally, the dataset includes detailed weather data corresponding to the collection dates, providing insights into environmental factors that may influence bacterial detection results.

## Contents
- **Establishment Information**
  - `EstablishmentID`: Identifier for the poultry establishment
  - `EstablishmentNumber`: Number assigned to the establishment
  - `EstablishmentName`: Name of the establishment
  - `State`: State where the establishment is located
- **Project Information**
  - `ProjectCode`: Code related to the sampling project
  - `ProjectName`: Name of the project
- **Sample Information**
  - `CollectionDate`: Date when the sample was collected
  - `SampleSource`: Source description of the sample
  - `SalmonellaSPAnalysis`: Salmonella analysis results (Positive/Negative)
  - `CampylobacterAnalysis1ml`: Campylobacter analysis results for 1ml sample
- **Weather Data** (Corresponding to the collection date)
  - `MaxTemp_Day0`, `MinTemp_Day0`, `AverageTemp_Day0`: Temperature data
  - `Humidity_Day0`, `Precipitation_Day0`: Humidity and precipitation
  - `WindSpeed_Day0`, `WindDirection_Day0`: Wind speed and direction
- **Weekday Data**
  - `Weekday`: Day of the week when the sample was collected

## Data Source
This dataset is provided by the USDA's Food Safety and Inspection Service (FSIS). All data have been collected under strict quality control and assurance procedures to ensure their accuracy and reliability.

## Usage
This dataset is intended for researchers and professionals in food safety, public health monitoring, and environmental science. It allows for the analysis of bacterial contamination in raw chicken and understanding how various environmental factors might impact such results.


