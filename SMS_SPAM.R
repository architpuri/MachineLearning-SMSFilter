#Installing required packages
#install.packages("gmodels") #for confusion Matrix
#install.packages("e1071") # naive Bayes
#install.packages("wordcloud") # WordCloud Presentation
#install.packages("tm") #for text mining (cleaning)
#install.packages("SnowballC") #wordStem to decrease no of words

# Importing the data
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE) #read data
str(sms_raw) #analyze structure
sms_raw$type <- factor(sms_raw$type)#convert to factors
str(sms_raw$type)
table(sms_raw$type)#to get table for ham/spam

# Using "tm" library (TEXT MINING)
#Text Data Preparation
library(tm)
sms_corpus <- VCorpus(VectorSource(sms_raw$text)) # to make corpus
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # To remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # To remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # To remove punctuation
removePunctuation("hello.world")
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello.world")

# Using "SnowballC" library for text data preparation
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # To eliminate unneeded whitespace
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

#rows indicate text messages & columns indicate words
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# Using "wordcloud" library ( WordCloud is a package for visualizing text data. 
# The larger Bold words represented occur more frequently whereas the smaller less Bold words do not appear as often.)

library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

# subset the training data into spam and ham groups
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# Frequent word indicators
sms_dtm_freq_train <- removeSparseTerms(sms_dtm_train, 0.999)# To remove outliers
sms_dtm_freq_train
findFreqTerms(sms_dtm_train, 5)
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)# taking words with 5 or more then 5 frequency
str(sms_freq_words)
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)
# margin=2 for columns 

# Training the model
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)


sms_test_pred <- predict(sms_classifier, sms_test)

# Evaluating and improving the performance of the model
library(gmodels)
#confusion matrix
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
#confusion matrix , laplace smoothening to handle new words.
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels, prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))

