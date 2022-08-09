# 1) INTRODUCTION

# 1.1) Objective
#This document creates a prediction system using root mean square error
#as a guideline to achieve the maximum score for the Recommendation
#System Capstone of the Harvardx Data Science course using the MovieLens
#10M dataset. The final model would succeed if the RMSE is under "0.8649"
#and the presented code in this document surpasses this goal by a fair
#amount, as you see follows.

# 1.2) Terminology
#Another critical thing to consider is how this document is presented.
#This document was written to make it easier to understand by colleagues
#with the same level of experience the writer has at this point. This
#decision was driven by the fact that the exercise explicitly informs us
#there will be a peer review, and the terminology follows this idea.

# 2) ANALYSIS

# 2.1) PREPARING THE DATA

# 2.1.1) Installing the packages
#We will need tools to achieve our goal. They need to be installed and
#loaded to be called when we need them. We verify if we already have them
#and install them if they are not present.

#IMPORTS
##Harvardx 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

##Others
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem")
if(!require(knitr)) install.packages('knitr', dependencies = TRUE)
if(!require(RColorBrewer)) install.packages("RColorBrewer")
if(!require(ggrepel)) install.packages("ggrepel")
tinytex::install_tinytex()

#Libraries
library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(lubridate)
library(ggplot2)
library(dplyr)
library(recosystem)
library(RColorBrewer)
library(ggrepel)

# 2.1.1) Creating the datasets
#This code creates the edx set and the validation set. The edx set will
#be divided into a train set and a test set, and both will be used to
#create and test the code. After we are sure it is working correctly, we
#can compare it with the validation set to guarantee we accomplished the
#task.

#When this code was written, the "MovieLens 10M" dataset could be
#downloaded through https://files.grouplens.org/datasets/movielens/ml-10m.zip).
#Even if it changed places, the logic behind what has been done does not
#change.

# Downloads the Movielens
dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Extracts the data using :: as reference
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Creates the data splitting when it finds a ::
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

# Creates the columns names
colnames(movies) <- c("movieId", "title", "genres")

# Creates the dataframe movies already definning the column type of data
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Ensures we have the same seed (and dataset) as everyone else 
set.seed(1, sample.kind="Rounding") 

# 10% of the whole set becomes the edx set
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clean space
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# 2.2) EXPLORATION & INSIGHTS
#Observing the datasets is essential to get insights into what we got and
#what we can create to predict better.

# 2.2.1) Heads
#The "head" command shows us the six rows from the top of each set to
#find if they have the same names, column order, and data types.

head(edx)
head(validation)

# 2.2.2) Glimpse
#Glimpse helps us pay attention to the column's dimensions and variable
#type.

glimpse(edx)
glimpse(validation)

# 2.2.3) Summary
#The summary shows the Quartiles, the Minimum, and Maximum values. It is
#crucial since it can show outliers not apparent using other approaches.

summary(edx)
summary(validation)

# 2.3) GRAPHS
# 2.3.1) Comparison: ratings x userid

#Creates a table containing how many times each userId repeats.
sorted_userId <- edx %>% 
group_by(userId) %>%
count() %>%
arrange(n) 

#Creates a histogram that divides the values between 50 bins.
ggplot (data = sorted_userId ) +
geom_histogram(mapping = aes(x=n, fill="gears", col=I("white")), bins=50) + 
theme_gray() +
theme(legend.position="none") +
scale_x_log10(breaks=c(1,5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480,40960)) +
labs(
    title = "How many times a user rated movies?",
    subtitle = "3.4.1) RATINGS x USERID",
    caption = "Dataset: edx only",
    x = "Number of ratings",
    y = "Number of users")

#There is a histogram approach to verify whether users rated many movies
#or not. The graph shows that most users rate a low number of movies, but
#the data starts at ten ratings per user.

# 2.3.2) Comparison: ratings x movieid

#Creates a table containing how many times each movieId repeats.
sorted_movieId <- edx %>% 
group_by(movieId) %>%
count() %>%
arrange(n)

#Creates a histogram that divides the values between 50 bins.
ggplot (data = sorted_movieId ) +
geom_histogram(mapping = aes(x=n, fill="gears", col=I("white")), bins=50) +
theme_gray() +
theme(legend.position="none") +
scale_x_log10(breaks=c(1,5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480,40960)) +
labs(
    title = "How many times a movie was rated?",
    subtitle = "3.4.2) RATINGS x MOVIEID",
    caption = "Dataset: edx only",
    x = "Number of ratings",
    y = "Number of movies")

#Some movies have almost no ratings, and some have many. It is vital to
#reduce the outliers' weight.

# 2.3.3) Comparison: ratings x genres

#Create a table that breaks the nested genres separated by | and counts 
#how many times each genre repeats.
sorted_genres <- edx %>% 
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  count() %>%
  arrange(n)

#Prevents scientific notation.
options(scipen=999) 

#Create labels using M or K to shorten the numbers,
#then creates sorted columns for each genre.
sorted_genres %>%
  mutate(genres = factor(genres, levels = genres)) %>%
  mutate(labels = ifelse(n>999999,paste(round(n/1000000,1),"M"), 
                         paste(round(n/1000,0),"K"))) %>%
  ggplot (mapping = aes(y=n, x=reorder(genres, -n), fill="gears", col=I("white"))) +
  geom_bar(stat="identity") +
  theme_gray() +
  labs(
    title = "How many times each genre of movie was rated?",
    subtitle = "3.4.3) RATINGS x GENRES",
    caption = "Dataset: edx only",
    x = "Number of Ratings by each Genre",
    y = "Genres") +
  theme(legend.position="none",
        axis.text.x = element_text(angle=90), axis.text.y = element_blank()) +
  geom_text(aes(label = labels, fontface = "bold", colour="#F8766D"), vjust=-0.5, size=2.5)

#There is a high discrepancy between movie genres. Regularization is
#needed.

# 2.4) CLEANING & MANIPULATING THE DATA
#There is a need to create new columns, and it is more productive to do
#them now using the edx dataset before splitting the test and train
#dataset.

# 2.4.1) Release Year (rey)
#It is noticeable that the title and the year of release are together,
#and it may be helpful to use them separately. The zeitgeist may impact
#the ratings, and it is preferred to use two extracts to assure the
#process was outputting what was idealized. This assures the finish,
#making the regex more readable and the years become detached from the
#title.

# Extracts the release year from the title 
release_year <- edx$title %>%
  str_extract(.,".{6}$") %>% #Extract the last 6 values
  str_extract(.,"(?<=\\().+?(?=\\))") %>% #Remove parenthesis
  as.integer() #Change it to a integer value

# Add the result as a column
edx <- as.data.frame(cbind(edx, release_year)) #Add the result as a new column

# 2.4.2) Rating Year (ray)
#If the Zeitgeist is a reasonable thought on handling the problem, it is
#natural to have not only the release_year but the rating_year too.

rating_year <- as.integer( #Assures the year is an integer
  year( #Extracts the year
    as.Date( #Extracts the date
      as.POSIXct( #R default time system.
        edx$timestamp,
        origin="1972-01-01")))) #The highest timestamp on the dataset. Unix Epoch
edx <- as.data.frame(cbind(edx, rating_year)) #Add the result as a new column

# 2.5) CREATING TRAIN & TEST SET

# 2.5.1) Partitioning
#The edx dataset will be split into train_set and test_set to reduce the
#chance of over-fitting when finding the best final model.

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating,
                                  times = 1,
                                  p = 0.2,
                                  list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# 2.5.2) Summary & Glimpse (Train & Test)
#The summary ensures the datasets are as intended.

summary(train_set)
glimpse(train_set)

summary(test_set)
glimpse(test_set)

# 2.6) METHODOLOGY

# 2.6.1) RMSE Function
#Even thou the ratings are factors (from 0.5 to 5, incrementing by 0.5
#each step), the exercise requires an RMSE approach. There will be a
#function created to calculate when it becomes needed.

##Creating a function to RMSE
RMSE <- function(reality, prediction){
  sqrt(mean((reality - prediction)^2))}

# 2.6.2) Mean
#Every time a new model is run, it is needed to know if there are
#improvements or not. The most basic model will always predict the
#average and will be used to create more accurate and complex models.

plain_mean <- mean(edx$rating)
plain_mean

# 2.6.3) Naive RMSE
#Naive RMSE determines how much a model fails when we guess the average
#value for every situation. If any model fails more than the Naive RMSE,
#something is wrong, and there is a need to start over. If the current
#model accomplishes a result very near to it, the model is not worth
#running due to the computational processing it uses. If it performs
#better than Naive RMSE, it is a step in the right direction.

naive_rmse <- RMSE(validation$rating, plain_mean)
naive_rmse

# 2.6.4) Linear model & Regularization
#A linear model was created where each predictor groups data then the
#error between the whole dataset mean and the actual rating is stored.
#Like most real datasets, there are outliers in each group, and it is
#needed to diminish their impact on the final model. Considering that the
#average is the sum of values divided by the count of entries, a way to
#regulate is to enhance by how much the sum is divided. This way, the now
#regulated average of a group with thousand entries will not be much
#impacted, but in a group with a low number of entries, the value will
#diminish and impact RMSE by very little. The idea is that larger groups
#are more reliable than small groups while we calculate central
#tendencies(mean, median, mode). Multiple loops are run trying
#incremental values, testing RMSE each time a loop is finished. This is
#done to find the correct arbitrary divisor value(lambda) for each group.
#An excessively high or low lambda value would worsen the RMSE, not
#improve it, so this is a long but necessary step.

rmse<-data.frame()
for(j in 1:5){ #The code is going to run 5 times with...
  var_group_by <- ifelse(j==1, "userId",
                         ifelse(j==2, "movieId",
                                ifelse(j==3, "genres",
                                       ifelse(j==4, "release_year", 
                                              "rating_year"))))
  for(i in 0:30){ #...31 times for each j loop.
    var_lambda <-i
    
    temp_set <- train_set %>% #It creates a temp_set...
      group_by_at(var_group_by) %>% #...grouped by the name of the group... 
      summarize(stray = sum(rating - plain_mean)/(n()+var_lambda)) 
    #..and calculates with regulation.
    
    #After this we join the values by the group...
    stray_set <- test_set %>% 
      left_join(temp_set, by=var_group_by) %>%
      select(-rating,-title,-genres,-release_year,-rating_year)
    
    #...then we create a set with the prediction using the mean and the error as reference.
    pred_set <- stray_set %>% 
      mutate(y = ifelse((plain_mean + stray)>5, 5, 
                        ifelse((plain_mean + stray)<0.5,0.5,
                               plain_mean + stray))) %>%
      select(-stray)
    
    #This create a organized dataset of result so we can compare.
    alocate <- nrow(rmse)+1
    rmse[alocate,1] <- var_group_by
    rmse[alocate,2] <- var_lambda
    rmse[alocate,3] <- RMSE(test_set$rating, pred_set$y)
  }
}
names(rmse) <- c("predictor","lambda","result")

#Now that we have every combination the code below shows us the best lambda
#in each group so we can make our decision.
best_reg <- rmse %>% 
  group_by(predictor) %>%
  slice(which.min(result)) %>%
  arrange(result)

lambda_u <- best_reg  %>% 
  filter(predictor=="userId") %>%
  .$lambda

lambda_m <- best_reg  %>% 
  filter(predictor=="movieId") %>%
  .$lambda

lambda_g <- best_reg  %>% 
  filter(predictor=="genres") %>%
  .$lambda

lambda_rey <- best_reg  %>% 
  filter(predictor=="release_year") %>%
  .$lambda                    

lambda_ray <- best_reg  %>% 
  filter(predictor=="rating_year") %>%
  .$lambda     

best_reg 

#It is noticeable that release_year(rey) and rating_year(ray) will
#probably not be helpful because they got the most prominent possible
#lambda and the worse RMSE by themselves. This points to the fact that
#they lack explicative power and should not be used in our final model.

reg_rmse <- data.frame()
#Creates the prediction based on each predictor and its best lambda.

#Using only userId
by_userId <- train_set %>% 
  group_by(userId) %>% 
  summarize(stray_u = sum(rating - plain_mean)/(n()+lambda_u))
#Using only movieId
by_movieId <- train_set %>% 
  group_by(movieId) %>% 
  summarize(stray_m = sum(rating - plain_mean)/(n()+lambda_m))
#Using only genres
by_genres <- train_set %>% 
  group_by(genres) %>% 
  summarize(stray_g = sum(rating - plain_mean)/(n()+lambda_g))
#Using only rey
by_release_year <- train_set %>% 
  group_by(release_year) %>% 
  summarize(stray_rey = sum(rating - plain_mean)/(n()+lambda_rey))
#Using only ray
by_rating_year <- train_set %>% 
  group_by(rating_year) %>% 
  summarize(stray_ray = sum(rating - plain_mean)/(n()+lambda_ray))

#Joins the sets and gets only by how much we failed.
stray_set <- test_set %>%
  left_join(by_userId, by="userId") %>%
  left_join(by_movieId, by="movieId") %>%
  left_join(by_genres, by="genres") %>%
  left_join(by_release_year, by="release_year") %>%
  left_join(by_rating_year, by="rating_year") %>%
  select(-rating,-title,-genres,-release_year,-rating_year)

#Creates a prediction to each mixed possibility of predictor using the 
#information above but it never let it goes under 0.5 or above 5.
pred_set <- stray_set %>% 
  
  mutate(y_u_m = ifelse(
    (plain_mean + stray_u + stray_m)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m)<0.5,0.5,
      plain_mean + stray_u + stray_m))) %>%
  
  mutate(y_u_m_g = ifelse(
    (plain_mean + stray_u + stray_m + stray_g)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m + stray_g)<0.5,0.5,
      plain_mean + stray_u + stray_m + stray_g))) %>%
  
  mutate(y_u_m_rey = ifelse(
    (plain_mean + stray_u + stray_m + stray_rey)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m + stray_rey)<0.5,0.5,
      plain_mean + stray_u + stray_m + stray_rey))) %>%
  
  mutate(y_u_m_ray = ifelse(
    (plain_mean + stray_u + stray_m + stray_ray)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m + stray_ray)<0.5,0.5,
      plain_mean + stray_u + stray_m + stray_ray))) %>%
  
  mutate(y_u_m_rey_ray = ifelse(
    (plain_mean + stray_u + stray_m + stray_rey + stray_ray)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m + stray_rey + stray_ray)<0.5,0.5,
      plain_mean + stray_u + stray_m + stray_rey + stray_ray))) %>%
  
  mutate(y_u_m_g_rey = ifelse(
    (plain_mean + stray_u + stray_m + stray_g + stray_rey)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m + stray_g + stray_rey)<0.5,0.5,
      plain_mean + stray_u + stray_m + stray_g + stray_rey))) %>%
  
  mutate(y_u_m_g_ray = ifelse(
    (plain_mean + stray_u + stray_m + stray_g + stray_ray)>5,5,
    ifelse((plain_mean + stray_u + stray_m + stray_g + stray_ray)<0.5,0.5,
           plain_mean + stray_u + stray_m + stray_g + stray_ray))) %>%
  
  mutate(y_u_m_g_rey_ray = ifelse(
    (plain_mean + stray_u + stray_m + stray_g + stray_rey + stray_ray)>5,5,
    ifelse(
      (plain_mean + stray_u + stray_m + stray_g + stray_rey + stray_ray)<0.5,0.5,
      plain_mean + stray_u + stray_m + stray_g + stray_rey + stray_ray))) %>%
  
  mutate(average = plain_mean) %>%
  select(-stray_u, -stray_m, -stray_g, -stray_rey, -stray_ray, -timestamp)

#Calculates the RMSE with each mixed probability and creates an organized table 
#where we can observe the results.
for(i in 3:ncol(pred_set)){
  alocate <- i-2
  reg_rmse[alocate,1] <- names(pred_set)[i]
  reg_rmse[alocate,2] <- RMSE(test_set$rating, pred_set[,i])
}
colnames(reg_rmse) <- c("method","rmse")

reg_rmse %>%
  arrange(rmse)

#Even though the best model(y_u\_m) using the linear model is far better
#than Naive RMSE, it is not enough to achieve the proposed exercise goal.
#The answer lay in Matrix Factorization.

# 2.6.5) Matrix Factorization -  Recosystem

#The MovieLens Grading Rubric stated:
#Note that to receive full marks on this project (...) your work
#on this project needs to build on code that is already provided
#The FAQ stated:
#Q: Can I use the recommenderlab, recosystem, etc. packages
#for my MovieLens project?
#A: If you understand how they work, yes!
  
#Recosystem is a package that utilizes parallel matrix factorization to
#solve this problem. It has built-in variables not only to find the best
#tunning parameters (dim, costp_l2, costq_l2, costp_l1, costq_l1, lrate)
#but to create the iterations and cross-validation by itself (niter,
#nfold) and even choose how many threads of parallel computing are going
#to be used (nthread) to accomplish the goal.
#LINK:
#https://cran.r-project.org/web/packages/recosystem/recosystem.pdf

# The https://github.com/yixuan/recosystem states:
# The data file for training set needs to be a
# sparse matrix with each line containing 03 numbers
# with the names user_index, item_index and rating.
# data_memory(): Specifies a data set from R objects

train_data <-  with(train_set, 
                    data_memory(user_index = userId,
                                item_index = movieId, 
                                rating = rating))
test_data  <-  with(test_set, 
                    data_memory(user_index = userId, 
                                item_index = movieId, 
                                rating = rating))

# Create the model object.
r <-  recosystem::Reco()

# Select the minimum and maximum tuning parameters
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), #default
                                       lrate = c(0.05, 0.1, 0.2), #default
                                       nthread = 3, niter = 10)) #default

# Train the algorithm using the optimal value ($min) of the tunning parameters above. 
r$train(train_data, 
        opts = c(opts$min, 
                 nthread = 3,
                 niter = 10))

pred_reco <- r$predict(test_data, out_memory())
RMSE(test_set$rating, pred_reco)

#As can be seen, the recosystem achieved the goal when compared to the
#test_set even with the default parameters.
#Even though it is a good indicator of the right direction, success with
#the test_set does not guarantee the same results using validation and
#the edx set. If the parameter creates some over-fitting, they will need
#to adjust.

edx_data <-  with(edx, 
                  data_memory(user_index = userId,
                              item_index = movieId, 
                              rating = rating))
val_data  <-  with(validation, 
                   data_memory(user_index = userId, 
                               item_index = movieId, 
                               rating = rating))

# Create the model object.
r <-  recosystem::Reco()

# Select the minimum and maximum tuning parameters
opts <- r$tune(edx_data, opts = list(dim = c(10, 20, 30), #default
                                     lrate = c(0.05, 0.1, 0.2), #default
                                     nthread = 3, niter = 10)) #default

# Train the algorithm using the optimal value ($min) of the tunning parameters above. 
r$train(edx_data, 
        opts = c(opts$min, 
                 nthread = 3,
                 niter = 10))

pred_reco <- r$predict(val_data, out_memory())
RMSE(validation$rating, pred_reco)

#As can be seen, the recosystem achieved the goal compared to the
#validation.

# 3) RESULTS

line <- nrow(reg_rmse)+1
reg_rmse[line,1] <- "Matrix Factorization"
reg_rmse[line,2] <- RMSE(validation$rating, pred_reco)

reg_rmse %>%
  arrange(rmse) 

#The matrix factorization clearly accomplished success (`<0.8649`) by a
#fair margin and beats every single method based on linear models.
#Recosystem is a powerful tool that can solve this problem even with
#default parameters.

# 4) CONCLUSION

#The presented solution is the result of many tries that are not fully
#described here because most crashed during execution or had such
#long-running times that it became impossible to proceed even when using
#any of the specialized cloud-based services that were affordable by the
#writer. Probably there are more elegant ways of accomplishing what was
#done here with more knowledge, experience, and computational power at
#disposal. Changing the Recosystem parameter would probably enhance the
#result even more if more time was at its disposal and could be
#considered for future iterations.