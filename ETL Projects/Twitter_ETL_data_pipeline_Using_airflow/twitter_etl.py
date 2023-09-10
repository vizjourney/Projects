import tweepy
import pandas as pd

def authenticate_twitter():
    access_key = ""  # Your API keys from Twitter development account
    access_secret = ""  # Your API secret keys from Twitter development account
    consumer_key = ""  # Your access token keys from Twitter development account
    consumer_secret = ""  # Your access token secret keys from Twitter development account

    auth = tweepy.OAuthHandler(access_key, access_secret)
    auth.set_access_token(consumer_key, consumer_secret)
    return tweepy.API(auth)

def get_user_tweets(api, username, max_tweets=200):
    tweets = api.user_timeline(screen_name=username, count=max_tweets, include_rts=False, tweet_mode='extended')

    refined_tweets = []
    for tweet in tweets:
        refined_tweet = {
            'user': tweet.user.screen_name,
            'text': tweet.full_text,
            'favorite_count': tweet.favorite_count,
            'retweet_count': tweet.retweet_count,
            'created_at': tweet.created_at
        }
        refined_tweets.append(refined_tweet)

    return pd.DataFrame(refined_tweets)

def main():
    api = authenticate_twitter()
    username = '@elonmusk'  # Replace with the Twitter handle you want to analyze
    df = get_user_tweets(api, username)
    df.to_csv('refined_tweets.csv', index=False)

if __name__ == "__main__":
    main()
