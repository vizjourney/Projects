# Twitter automatic ETL pipeline with apache airflow

This repository contains two Python scripts: `twitter_etl.py` and `twitter_dag.py`, which are used to extract and refine tweets from a Twitter user's timeline and create an Airflow DAG for scheduling the ETL process, respectively.

## Twitter ETL Script

### Prerequisites

Before running the Twitter ETL script, make sure you have the following:

1. Twitter Developer Account: You need to create a Twitter Developer Account and obtain API keys and access tokens.

2. Python Environment: Ensure you have Python installed on your system.

### Setup

1. Clone the repository to your local machine.

2. Install the required packages by running the following command:

   ```bash
   pip install -r requirements.txt


## Mannual

1. Fill in your Twitter API keys and access tokens in the script.
Usage

2. Run the script by executing the following command:

python twitter_etl.py

3. The script will extract tweets from the specified Twitter user's timeline, refine the data, and save it in a CSV file called refined_tweets.csv.

## Customization
You can customize the script by changing the Twitter username in the main() function and adjusting the number of tweets to fetch.

### Twitter DAG Script

1. The Twitter DAG script allows you to create an Airflow DAG for scheduling the ETL process.

2. Prerequisites
    Before running the Twitter DAG script, make sure you have the following:

    1. Apache Airflow Installed: You need to have Apache Airflow installed on your system.
## Usage
1. Ensure Apache Airflow is correctly set up and configured.
2. Place the twitter_etl.py script in your Airflow DAGs directory.
3. Import the DAG from twitter_dag.py in your Airflow DAG configuration.
4. Customize the DAG as needed, specifying the schedule interval and other parameters.
5. Run Apache Airflow to schedule and execute the ETL process.