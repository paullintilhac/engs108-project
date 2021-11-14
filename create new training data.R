library(data.table)
dat = read.csv("~/engs108-project/data/dow_30_2009_2020.csv")
smallDat = dat[dat$tic %in% c("AXP","BA"),]

write.csv(smallDat,file = "~/engs108-project/data/dow_30_2009_2020_small.csv",column.names=FALSE,row.names=FALSE,quote=FALSE)
?write.csv
head(dat$tic)
