# This data set from the KAGGLE website which provider of varies datasets. This data set basically belongs to the people who are bank churners. Here we have 10127 examples and 23 features of the data set. But after cleansing the data we can setup the data set with 7081 examples and 20 features (with response variable).
# Here I tried to predict the client income category( more than $60K & less than $60K) with using  some different classification machine klearning algorithms such as,
# K Nearest Neighbors (KNN)
# Naive Bayes (NB)
# Decision Tree (DT)
# Random Forest (RF)

#Let's begin by loading the required, needed libraries and importing the BankChurners.csv datasheet.
library(tidyverse)
library(class)
library(caret)
library(e1071)
library(C50)
library(randomForest)

df <- read.csv("BankChurners.csv", stringsAsFactors = TRUE)

# Cleaning the data set
#-- There is some unknown data of the data set. For performing good accurate algorithms we have to remove those. Even though we lose good features, it doesn't matter we have a big data collection. As well as we have to remove some columns which are collected as not valuable for this models.
df <- df %>% select(-1, -22, -23)
df <- df %>% filter(!Education_Level == "Unknown") %>%
             filter(!Marital_Status == "Unknown") %>%
             filter(!Income_Category == "Unknown")
str(df)

#-----------------------------------------------------------------------------
####### Visualization some basic categorical features

df %>% ggplot(aes(x = Attrition_Flag, fill = Attrition_Flag)) +
  geom_bar() +
  scale_y_continuous(breaks = seq(0, 6000, by = 1000)) +
  labs(title = "Clients of attrition", 
       y = "Number of clients") 
# -- Just looking at the type of attrition we can see existing clients are very biggr than attrited clients.

Hmisc::describe(df[2:3])
# -- Here we can observed nearly same portion of gender involve this data collection and the male category is high some little bit 
# -- By looking at the clients age we can see there are 70+ persons and there are no persons who not below the 25 years old. The mean average age is 46 years.

df %>% ggplot(aes(x = Education_Level, fill = Gender)) +
  geom_bar() + facet_wrap(~ Gender) +
  labs(title = "Education levels accordingto the Gender", 
       y = "Number of clients") +
  coord_polar() +
  theme_grey() 
# -- There are most client has a degree. There are few of doctors and post graduate.
  
df %>% ggplot(aes(x = Marital_Status, fill = Dependent_count)) +
  geom_bar() + facet_wrap(~ Dependent_count) +
  labs(title = "Marital status and their dependents", 
       y = "Number of clients") +
  coord_flip() +
  scale_y_continuous(breaks = seq(0, 1000, by = 100)) +
  theme_grey() 
# -- We can see high evaluation of dependents have for the married clients which are 2 and 3 dependents.

Hmisc::describe(df[8])
# -- Most people have Blue cards and few of have Platinum cards as 0.002 proportion from the 7081 clients.


#-------------------------------------------------------------------------------
###### exploring and preparing the data
#-- For the get sensitive features here we transform all the categories into the numeric format. Then we can use them also as our modeling features.

# We recode the data as 1 for "Existing Customer" & 0 for "Attracted Customer".
df$Attrition_Flag <- ifelse(df$Attrition_Flag == "Existing Customer", 1, 0)

# We recode the data as 1 for "Male" & 0 for "Female".
df$Gender <- ifelse(df$Gender == "M", 1, 0)

# We recode here as 1 - "Doctorate", 2 -  "Graduate", 3 - "Post-Graduate", 4 - "High School", 5 - "College" & 0 for "Uneducated".
df$Education_Level <- ifelse(df$Education_Level == "Doctorate", 1,
                             ifelse(df$Education_Level == "Graduate", 2,
                                    ifelse(df$Education_Level == "Post-Graduate", 3,
                                           ifelse(df$Education_Level == "High School", 4,
                                                  ifelse(df$Education_Level == "College", 5,0)))))

# Recode here as 1 - "Married", 2 - "Single" & 0 for "Divorced". 
df$Marital_Status <- ifelse(df$Marital_Status == "Married", 1,
                            ifelse(df$Marital_Status == "Single", 2, 0))

# Here the output variable and it has five different categories. We recode those into two categories. 
# Which as earning the income more than $60K and less than $60K, 
# 0 - less than $60K
# 1 - more than $60K
df$Income_Category <- ifelse(df$Income_Category == "$120K +", 1,
                             ifelse(df$Income_Category == "$80K - $120K", 1,
                                    ifelse(df$Income_Category == "$60K - $80K", 1, 0)))

# Here we recode the card categories as 
# 1 - Platinum
# 2 - Gold
# 3 - Silver
# 4 - Blue
df$Card_Category <- ifelse(df$Card_Category == "Platinum", 1,
                           ifelse(df$Card_Category == "Gold", 2,
                                  ifelse(df$Card_Category == "Silver", 3,4)))

# Let’s see the how response feature categorize as getting income as more than $60K and less than $60K
round(prop.table(table(df$Income_Category)) * 100, digits = 1)
# -- As nearly 60% of the clients have the income less than the $60K.


# -----------------------------------------------------------------------------------
####### Performing Machine Learning Algorithms

# Segregating data into training and testing perposes 
set.seed(1)
intrain <- createDataPartition(df$Income_Category, p = 0.75, list = FALSE)
training <- df[intrain, ]
testing <- df[-intrain, ]

# Transformation & normalizing data
normalize <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))
}

train_N <- as.data.frame(lapply(training[-7], normalize)) 
test_N <- as.data.frame(lapply(testing[-7], normalize)) 

train_Lab <- as.factor(training$Income_Category)
test_Lab <- as.factor(testing$Income_Category)

# ---------------------------------------------------------------------------------
#### K Nearest Neighbors (KNN)

# Finding best K value
k_values <- c(1:100)
accuracy <- vector("double", length(k_values))
for (i in seq_along(k_values)) {
  model <- knn(train = train_N, test = test_N, cl = train_Lab, k = i)
  q <- confusionMatrix(model, test_Lab)
  accuracy[[i]] <- q[["overall"]][["Accuracy"]]*100
}

acc_table_N <- data.frame(k_values, accuracy)

# Plotting the variation
ggplot(data = acc_table_N) +
  geom_point(mapping = aes(x = k_values, y = accuracy), color = "red") +
  geom_line(mapping = aes(x = k_values, y = accuracy))+
  scale_y_continuous(breaks = seq(85, 89, by = 0.25)) +
  scale_x_continuous(breaks = seq(1,100, by = 4)) +
  labs(title = "Model accuracy variation with different K values")

# Highest accuracy best k values
head(arrange(acc_table_N, desc(accuracy)))

# Training the model according to the best K value & predicting using knn() function.
kNN_pred <- knn(train = train_N, test = test_N, 
                 cl = train_Lab, k = arrange(acc_table_N, desc(accuracy))[1,1])

# Evaluating model performance by confusionMatrix()
confusionMatrix(kNN_pred, test_Lab)

# Improving model performance with changing the feature values for standardization.
train_S <- as.data.frame(lapply(training[-7], scale)) 
test_S <- as.data.frame(lapply(testing[-7], scale)) 

# Finding best K value for standardization features.
accuracy <- vector("double", length(k_values))
for (i in seq_along(k_values)) {
  model <- knn(train = train_S, test = test_S, cl = train_Lab, k = i)
  q <- confusionMatrix(model, test_Lab)
  accuracy[[i]] <- q[["overall"]][["Accuracy"]]*100
}

acc_table_S <- data.frame(k_values, accuracy)

# Plotting the variation
ggplot(data = acc_table_S) +
  geom_point(mapping = aes(x = k_values, y = accuracy), color = "red") +
  geom_line(mapping = aes(x = k_values, y = accuracy))+
  scale_y_continuous(breaks = seq(85, 89, by = 0.25)) +
  scale_x_continuous(breaks = seq(1,100, by = 4)) +
  labs(title = "Model accuracy variation with different K values",
       subtitle = "with Standardization values")

# Highest accuracy best k values by standardization features.
head(arrange(acc_table_S, desc(accuracy)))

# Training the improved model & predicting by selected best K value.
kNN_pred_S <- knn(train = train_S, test = test_S, 
                 cl = train_Lab, k = arrange(acc_table_S, desc(accuracy))[1,1])

# Evaluating model performance
confusionMatrix(kNN_pred_S, test_Lab)


#--------------------------------------------------------------------------------------
#### Naive Bayes (NB)

# Training the model using naiveBayes() function & Predicting the values from training NB model.
nb_model <- naiveBayes(training[-7], train_Lab)
nb_pred <- predict(nb_model, testing[-7])

# Evaluating model performance
confusionMatrix(nb_pred, test_Lab)


#--------------------------------------------------------------------------------------
#### Decision Tree (DT)

# Training the model using C5.0() function & Predicting the values from training DT model.
dT_model <- C5.0(training[-7], train_Lab)
dT_pred <- predict(dT_model, testing[-7])

# Evaluating model performance
confusionMatrix(dT_pred, test_Lab)

# To get more accuracy from the model we have to tune the trail parameter.
# Finding best trial number for improving model performance by tuning trials parameter.
trials_values <- c(1:30)
dt_accuracy <- vector("double", length(trials_values))
for (i in seq_along(trials_values)) {
  dT_model <- C5.0(training[-7], train_Lab, trials = i)
  dT_pred <- predict(dT_model, testing[-7])
  q <- confusionMatrix(dT_pred, test_Lab)
  dt_accuracy[[i]] <- q[["overall"]][["Accuracy"]]*100
}

acc_table_DT <- data.frame(trials_values, dt_accuracy)

# Plotting the variation
ggplot(data = acc_table_DT) +
  geom_point(mapping = aes(x = trials_values, y = dt_accuracy), color = "green", size = 6) +
  geom_line(mapping = aes(x = trials_values, y = dt_accuracy))+
  scale_y_continuous(breaks = seq(85, 93, by = 0.25)) +
  scale_x_continuous(breaks = seq(1,30, by = 1)) +
  labs(title = "  DT Model accuracy variation with different trials values")

# Highest accuracy best trail values
head(arrange(acc_table_DT, desc(dt_accuracy)))

# Training & predicting improved model by selected best trail value.
dT_model <- C5.0(training[-7], train_Lab, 
                 trials = arrange(acc_table_DT, desc(dt_accuracy))[1,1])
dT_pred <- predict(dT_model, testing[-7])

# evaluating model performance
confusionMatrix(dT_pred, test_Lab)


#-------------------------------------------------------------------------------------
#### Random Forest (RF)

# WE can perform a good accuracy model by selecting good number of trees (here we used 500 trees) and setting best mtry value.
# Finding best mtry value for tune the parameters
mtry_values <- c(2:15)
rf_accuracy <- vector("double", length(mtry_values))
for (i in seq_along(mtry_values)) {
  rf_model <- randomForest(training[-7], train_Lab, ntree = 500, mtry = i)
  rf_pred <- predict(rf_model, testing[-7])
  q <- confusionMatrix(rf_pred, test_Lab)
  rf_accuracy[[i]] <- q[["overall"]][["Accuracy"]]*100
}

acc_table_RF <- data.frame(mtry_values, rf_accuracy)

# Plotting the variation
ggplot(data = acc_table_RF) +
  geom_point(mapping = aes(x = mtry_values, y = rf_accuracy), color = "cyan", size = 6) +
  geom_line(mapping = aes(x = mtry_values, y = rf_accuracy))+
  scale_y_continuous(breaks = seq(85, 93, by = 0.25)) +
  scale_x_continuous(breaks = seq(2,15, by = 1)) +
  labs(title = "  RF Model accuracy variation with different mtry values")

# Highest accuracy best mtry values 
head(arrange(acc_table_RF, desc(rf_accuracy)))

# Training & predicting improved model by selected best mtry value.
rf_model <- randomForest(training[-7], train_Lab, 
                         ntree = 500, mtry = arrange(acc_table_RF, desc(rf_accuracy))[1,1])
rf_pred <- predict(rf_model, testing[-7])

# Evaluating model performance
confusionMatrix(rf_pred, test_Lab)


# So finally we perform some different machine learning algorithms for predict the income group of the clients.
# Through the those algorithms we can see how the accuracy variation of those models.
# So we can create a summarize accuracy table for the all models as below.
models <- c("KNN", "Naive Bayes", "Decision Tree", "Random Forest")
accuracies <- c(confusionMatrix(kNN_pred_S, test_Lab)[["overall"]][["Accuracy"]]*100,
                confusionMatrix(nb_pred, test_Lab)[["overall"]][["Accuracy"]]*100,
                confusionMatrix(dT_pred, test_Lab)[["overall"]][["Accuracy"]]*100,
                confusionMatrix(rf_pred, test_Lab)[["overall"]][["Accuracy"]]*100)
(summary_acc_table <- data.frame("Models" = models, "Accuracy" = accuracies))

# Plotting the results from the summarize accuracy table.
summary_acc_table %>% ggplot(mapping = aes(x = Models, y = Accuracy)) + 
                               geom_col(fill = c("aquamarine2", "aquamarine1",
                                                 "aquamarine4", "aquamarine3")) +
  coord_cartesian(ylim = c(86, 91)) +
  scale_y_continuous(breaks = seq(86, 91, by = 0.5)) +
  labs(title = "Summarize accuracies of 4 Algorithms") +
  theme_linedraw()

#----------------------------------------------------------------------------------------
# Conclusion

# Then the finally as a result of my task, I can just tell to community the best machine learning algorithm for the bank churners for predicting the income category(as less than $60K and more than $60K)is the Decision Tree machine learning algorithm.



