### Case 2: Online music ####
lastfm <- read.csv(file.choose()) # choose lastfm.csv
str(lastfm) # 289955 obs. of  4 variables
lastfm[10:20,]
length(unique(lastfm$user)) # 15000 users
lastfm$user <- factor(lastfm$user) # from class "integer" to class "factor"
nlevels(lastfm$user) # 15000 levels (users)
nlevels(lastfm$artist) # 1004 artists (or bands)

library(arules) # for Association Rules package
playlist <- split(x=lastfm$artist,f=lastfm$user) ## split a vector into groups, wow ! It's a large list of 938.3 Mb
##### A litlle bit detour!
(x <- 1:10)
(y <- c("E", "D", rep("A",3), rep("D",3), "E", "E"))
split(x, y)
#####
str(playlist) # A large list, the artists each user bought !
playlist[1:2] # the first two listeners (1 and 3) listen to the following bands
musicrules <- apriori(playlist,parameter=list(support=.01,confidence=.5)) # only rules with support > 0.01 and confidence > .50 # Error in asMethod(object) : can not coerce list with transactions with duplicated items
# So, let's make a function to identify it first !

# Actually, an error will occur as u coerce "list" to "transactions"!
playlist <- as(playlist, "transactions")
# Error in asMethod(object) : can not coerce list with transactions with duplicated items

duplicatedArtist <- function(x) {
  length(x) != length(unique(x))
} # a function for identifying duplicatedArtist
sum(sapply(playlist, duplicatedArtist)) # 2
which(sapply(playlist, duplicatedArtist)) # 6980 (5290) & 9753 (7422)

##### A lit bit detour !
(x <- 10:1)
x < 3
which(x < 3) # return indexes where x is less than 3
#####
playlist[[5290]]
unique(playlist[[5290]])
playlist[[7422]]
unique(playlist[[7422]])
duplicated(playlist[[7422]]) # 17th artist (james brown) is same as the 8th artist (james brown)
playlist <- lapply(playlist,unique) ## remove artist duplicates
# if you skip this line, you will get "Error in asMethod(object): can not coerce list with transactions with duplicated items"
class(playlist) # class of "list

playlist <- as(playlist,"transactions") # from class "list" to "transactions" defined in {arules}
class(playlist) # class of "transaction" defined in package "arules" (S4 Object-Oriented programming)

playlist # S4 OO (object-oriented) more descent!!
inspect(playlist[1])
## view this as a list of "transactions"
## transactions is a data class defined in arules
# large transactions (15000 elements, 2.9 Mb)

dim(playlist)
length(playlist)
size(playlist) # size of 15000 transactions
tmp <- hist(size(playlist)) # slightly right-skewed dist.
tmp
hist(size(playlist), breaks=56) # 16*5=80, but 56 is enough!

itemFrequency(playlist)[10:15]
itemFrequency(playlist)
length(itemFrequency(playlist)) # 1004

itemFrequency(playlist, type="absolute")[10:15]
itemFrequency(playlist, type="absolute")
tmp <- hist(itemFrequency(playlist, type="absolute")) # heavily right-skewed dist.
hist(itemFrequency(playlist, type="absolute"), breaks=70) # 14*5=70 (One assumptions is items occur in the database following a, possibly unknown, but stable process and that the items occur in the database with roughly similar frequencies. Nevertheless, in the real world, transactions data have a frequency distribution highly skewed with almost items occurring in an infrequent way while just some of them occur with high frequency.)
### Reference: https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/arulesNBMiner

class(itemFrequency(playlist)) # a **named** numeric vector
sort(itemFrequency(playlist), decreasing=T)[1:10]

itemFrequencyPlot(playlist,support=.018,cex.names=1.5)
itemFrequencyPlot(playlist,support=.08,cex.names=1.5, topN=12) # which one is better?
?itemFrequencyPlot

op <- par(mar=c(8, 4, 1, 2) + 0.1)
barplot(sort(colSums(as(playlist,'matrix')), decreasing=TRUE)[1:15], las=2)
par(op) # 還原預設設定mar=c(5, 4, 4, 2) + 0.1)
?par

summary(playlist)

### frequent itemsets by {eclat}
freqItemsets <- eclat(playlist)
inspect(freqItemsets) # same as sort(itemFrequency(playlist), decreasing=T)[1:7]

### maximally frequent itemsets
maxItemsets <- eclat(playlist, parameter=list(target="maximally frequent itemsets"))
inspect(maxItemsets) # same as frequent itemsets

### closed frequent itemsets
closedItemsets <- eclat(playlist, parameter=list(target="closed frequent itemsets"))
inspect(closedItemsets) # same as frequent itemsets
##########################

class(playlist)
musicrules <- apriori(playlist,parameter=list(support=.01,confidence=.5)) # only rules with support > 0.01 and confidence > .50

musicrules # it's an S4 object!

inspect(musicrules) # 50 rules

quality(musicrules)

# association rules filtering
musicrules_small <- subset(musicrules, subset=lift > 5) # subset(原集合, subset = 子集條件)

inspect(musicrules_small)
inspect(sort(musicrules_small, by="confidence")) # 此sort非彼sort也
class(musicrules_small)
### Can you filter out redundnat rules and visulaize the final results?
library(arulesViz) # Viz: Visualization
plot(musicrules_small)
plot(musicrules_small, method="graph")
plot(musicrules_small, method="paracoord")
plot(musicrules_small, interactive = TRUE)