library(data.table)
library(splitstackshape)
library(caret)

# Read input data
docs_train <- fread('data/train.csv')
docs_test  <- fread('data/test.csv')
stopwords  <- fread('data/stopwords.txt', header = F)[[1]]

# Create data partitions for training and validation (balance the class distributions within the splits)
# Changed p to 1.0 training for submission, otherwise p = 0.7 was used during development
partition  <- createDataPartition(as.factor(docs_train$Book), p = 1.0)[[1]]
docs_train <- docs_train[partition]
docs_val   <- docs_train[!partition]
docs_test  <- docs_test
rm(partition)

setkey(docs_train, 'DocumentID')
setkey(docs_val, 'DocumentID')
setkey(docs_test,  'DocumentID')

# Split documents into terms and remove stopwords
split_into_terms <- function (data) {
    if (nrow(data) > 0) {
        data_split <- cSplit(data, 'Document', sep = ' ', direction = 'long')
        setnames(data_split, 'Document', 'Term')
        data_split[!(Term %in% stopwords)][!grepl('[0-9]+', Term)]
    } else {
        data.table(DocumentID = numeric(), Term = factor())
    }
}

docs_train_terms <- split_into_terms(docs_train)
docs_val_terms   <- split_into_terms(docs_val)
docs_test_terms  <- split_into_terms(docs_test)

# -----------------------------------------------------------------------------

# Count of total terms per book
book_terms <- docs_train_terms[, list(BookTerms = .N), by = list(Book)]
setkey(book_terms, 'Book')

# Count of each term per book
term_book_count <- docs_train_terms[, list(Count = .N), by = list(Term, Book)]
setkey(term_book_count, 'Book')

# Normalise counts because of unbalanced training data
term_book_count <- term_book_count[book_terms][, list(Term, Book, Count, CountNorm = Count / BookTerms)]

# Calculate distribution of terms across books
term_book_count <- term_book_count[, list(Book, Count, CountNorm, CountShare = CountNorm / sum(CountNorm)), by = list(Term)]

# -----------------------------------------------------------------------------

# Make prediction based on book distribution of terms
predict_book <- function (doc_terms, docs, test = F) {
    pred <- merge(doc_terms[, list(DocumentID, Term)], term_book_count, by = 'Term', allow.cartesian = T)
    pred <- pred[, list(Score = sum(CountShare, na.rm = T)), by = list(DocumentID, Book)]
    pred <- unique(pred[order(DocumentID, -Score)], by = 'DocumentID')[, list(DocumentID, Pred = Book, Score)]
    setkey(pred, 'DocumentID')
    setkey(docs, 'DocumentID')
    if (test) {
        pred[docs][, list(DocumentID, Pred)]
    } else {
        pred[docs][, list(DocumentID, Book, Pred)]
    }
}

pred_val_2  <- predict_book(docs_val_terms, docs_val)
pred_test_2 <- predict_book(docs_test_terms, docs_test, test = T)
write.csv(pred_test_2[, list(DocumentID, Predicted_class = Pred)], 'results/pred_book_2_termfreq.csv',
          row.names = FALSE, quote = FALSE)

# Show confusion matrix for validation data
if (nrow(pred_val_2) > 0) {
    print(confusionMatrix(pred_val_2$Book, pred_val_2$Pred))
}
