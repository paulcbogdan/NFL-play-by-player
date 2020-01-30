library(nflscrapR)
#library(xlsx)
preseason_pbp_2019_wk13_to_16 <- scrape_season_play_by_play(2019, type='reg', weeks=c(13, 14, 15, 16)) # c(1,2,3,4,5,6,7,8,9,10,11,12) c(13, 14, 15, 16)

write.csv(preseason_pbp_2019_wk13_to_16,"E:\\PycharmProjects\\FootballSim\\pbp_data\\2019_wk13141516.csv", row.names = FALSE)
