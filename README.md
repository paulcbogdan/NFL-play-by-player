# NFL-play-by-player
This code simulates NFL games in a play-by-play fashion. Heavily uses the Pandas and Statsmodels packages. The termcolor and tqdm packages are also used for aesthetics. See my reddit post: https://www.reddit.com/r/nfl/comments/ew9dog/simulating_the_2018_and_2019_nfl_seasons_one_play/

organize_data.py takes the original .xlsx spreadsheet detailing play-by-play information and converts it into a Pandas dataframe and does some very minimal pre-processing. Honestly, it may not be necessary. The original play-by-play data can be found here: https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016 For the 2019 season you will need to use this package used to create the Kaggle datset: https://github.com/maksimhorowitz/nflscrapR 

scraper.R is an R script used for downloading the data. Very basic. This play-by-play data is stored in pbp_data for use by organize_data.py. It begins by making .pkl files out of the dataspreadsheets, as they are much quicker to load.

game_interaction.py is the code where the game is simulated. This code will print things even if you set print=False. Setting print=True just makes it print even more things!

predictors.py manages the regressions employed by game_interaction.py. Most of the effort went into building those regressions which predict percentiles and then map those percentiles to a distribution, as described in the initial Reddit post. 

DATE_NUMBER_TO_WEEK_NUMBER_data.py has some dictionaries specifying which dates correspond to which NFL season weeks. Only done for the 2018 and 2019 seasons. Used by compile_historical_spreads.py
compile_historical_spreads.py gets matchup information for every week. Its input data should be stored in historical_odds and can be downloaded from here: https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm

mass_simulate_compare_odds.py takes this historical odds information and uses it to simulate games. results are saved in results_save

analyze_results.py analyzes the results in results_save. Around this point, I got pretty hasty in my code and it is the ugliest part of all of this

utils.py some basic things. I really like pickle_wrap. It it a way to automatically save the output of a function (passed as a Lambda) in a pickle file. If the pickle file already exists and easy_override=False, then the file is just loaded.
