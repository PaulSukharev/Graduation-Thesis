import sys
import csv
reload(sys)
sys.setdefaultencoding('utf8')

FILENAME = "datasetApple.csv"

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main():

	sinceYear = 2006
	untilYear = 2018
	twweetsPerDay = 200

	for year in range(sinceYear, untilYear):
		for month in range(1, 13):
			for day in range(1, 32):
				tweetCriteria = got.manager.TweetCriteria().setQuerySearch('AAPL').setSince(str(year) + "-" + str(month) + "-" + str(day)).setUntil(str(year) + "-" + str(month) + "-" + str(day + 1)).setMaxTweets(twweetsPerDay)
				with open(FILENAME, 'a') as csv_file:
					writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
					for tweet in got.manager.TweetManager.getTweets(tweetCriteria):
						writer.writerow([tweet.username, tweet.text, tweet.mentions, tweet.date])

	print("Writing complete")

if __name__ == '__main__':
	main()
