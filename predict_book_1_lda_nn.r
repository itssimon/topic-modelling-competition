library(data.table)
library(stm)
library(topicmodels)
library(ldatuning)
library(caret)

# Read input data
docs_train <- fread('data/train.csv')
docs_test  <- fread('data/test.csv')
stopwords  <- fread('data/stopwords.txt', header = F)[[1]]

# Pre-process training documents and convert to document-term matrix using stm package
corpus_train <- textProcessor(docs_train$Document, stem = FALSE, customstopwords = stopwords)
corpus_train <- prepDocuments(corpus_train$documents, corpus_train$vocab, corpus_train$meta, lower.thresh = 2)
corpus_train <- convertCorpus(corpus_train$documents, corpus_train$vocab, type = 'slam')

# Topic number selection using ldatuning package
# k <- FindTopicsNumber(corpus_train, topics = seq(from = 4, to = 120, by = 2),
#                       metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
#                       method = "Gibbs", control = list(alpha = 0.01, seed = 1234), mc.cores = 4L, verbose = TRUE)

# Train LDA model and create topic distribution data set
# A low alpha-value will lead to documents being less similar in terms of what topics they contain
lda_control <- list(alpha = 0.01, iter = 5000, seed = 1234, verbose = 100)
lda_model   <- LDA(corpus_train, k = 32, method = 'Gibbs', control = lda_control)
data_train  <- cbind(docs_train[, list(DocumentID, Book)],
                     data.frame(posterior(lda_model)$topics))

# -----------------------------------------------------------------------------

# Create data partitions for training and validation (balance the book distributions within the splits)
# Changed p to 1.0 training for submission, otherwise p = 0.7 was used during development
partition  <- createDataPartition(as.factor(data_train$Book), p = 1.0)[[1]]
data_val   <- data_train[!partition]
data_train <- data_train[partition]
rm(partition)

# Search for best hyper-parameters for the neuronal network model
# nn_train_control <- trainControl(method = 'cv', number = 5, allowParallel = TRUE, verboseIter = TRUE)
# nn_tune_grid <- expand.grid(size = 2:5, decay = c(0.1, 0.01, 0.001), bag = c(TRUE, FALSE))

# Fit a neuronal network model on topic distributions to predict book
# using tuned hyper-parameters
nn_train_control <- trainControl(method = 'none', verboseIter = TRUE)
nn_tune_grid <- expand.grid(size = 3, decay = 0.1, bag = TRUE)
nn_model <- train(Book ~ . - DocumentID, data = data_train, method = 'avNNet',
                  trControl = nn_train_control, tuneGrid = nn_tune_grid)

# Make predictions for validation set
pred_val_1 <- data_val[, list(DocumentID, Book, Pred = predict(nn_model, data_val))]

# Show confusion matrix
if (nrow(pred_val_1) > 0) {
    print(confusionMatrix(pred_val_1$Book, pred_val_1$Pred))
}

# -----------------------------------------------------------------------------

# Pre-process test documents and convert to document-term matrix
corpus_test <- textProcessor(docs_test$Document, stem = FALSE, customstopwords = stopwords)
corpus_test <- convertCorpus(corpus_test$documents, corpus_test$vocab, type = 'slam')

# Determine the posterior probabilities of the topics for test documents
data_test <- cbind(docs_train[, list(DocumentID)],
                   data.frame(posterior(lda_model, corpus_test, control = lda_control)$topics))

# Make and save predictions for test set
pred_test_1 <- data_test[, list(DocumentID, Pred = predict(nn_model, data_test))]
write.csv(pred_test_1[, list(DocumentID, Predicted_class = Pred)], 'results/pred_book_1_lda_nn.csv',
          row.names = FALSE, quote = FALSE)
