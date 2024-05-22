#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#define VOCAB_SIZE 1000
#define EMBEDDING_SIZE 100
#define WINDOW_SIZE 8
#define LEARNING_RATE 0.01
#define EPOCHS 200

typedef struct {
    int target;
    int context;
} Pair;

typedef struct {
    char word[256];
    double tfidf;
} Word;

void initialize_vectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE]);
double random_double();
void softmax(double* input, double* output, int size);
void train(Pair* pairs, int pair_count, int vocab_size);
Pair* generate_pairs(const char* filename, int* pair_count);
void split_sentences(const char* text, FILE* output_file);
void tokenize_words(const char* text, FILE* output_file);
int tokenize();
void save_vectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE], const char* filename);
void compute_tfidf(const char* filename, Word* words, int* word_count);
void save_tfidf(Word* words, int word_count, const char* filename);
Word find_max_tfidf_word(Word* words, int word_count);

int main() {
    printf("Reading input.txt\n");

    FILE* input_file = fopen("input.txt", "r");
    if (input_file == NULL) {
        perror("파일 fopen 오류");
        return 1;
    }
    printf("Reading input.txt complete\n");

    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    char* text = (char*)malloc(file_size + 1);
    if (text == NULL) {
        perror("메모리 malloc 오류");
        fclose(input_file);
        return 1;
    }

    fread(text, 1, file_size, input_file);
    text[file_size] = '\0';

    fclose(input_file);

    printf("Splitting sentences into output.txt\n");

    FILE* output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        perror("파일 fopen 오류");
        free(text);
        return 1;
    }
    Sleep(500);
    split_sentences(text, output_file);

    fclose(output_file);
    free(text);

    printf("Tokenizing sentences into tokenized.txt\n");

    if (tokenize() != 0) {
        printf("Tokenization failed.\n");
        return 1;
    }

    printf("Generating pairs from tokenized.txt\n");

    int pair_count;
    Pair* pairs = generate_pairs("tokenized.txt", &pair_count);

    if (pairs == NULL) {
        return 1;
    }

    printf("Training Skip-Gram model %d: \n", pair_count);

    train(pairs, pair_count, VOCAB_SIZE);

    free(pairs);

    printf("Computing TF-IDF values\n");

    Word words[VOCAB_SIZE];
    int word_count = 0;
    compute_tfidf("tokenized.txt", words, &word_count);
    save_tfidf(words, word_count, "weight.txt");

    Word max_word = find_max_tfidf_word(words, word_count);
    printf("Word with highest TF-IDF: %s (%.6f)\n", max_word.word, max_word.tfidf);

    return 0;
}

void initialize_vectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE]) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            vectors[i][j] = (random_double() - 0.5) / EMBEDDING_SIZE;
        }
    }
}

double random_double() {
    return (double)rand() / (double)RAND_MAX;
}

void softmax(double* input, double* output, int size) {
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void train(Pair* pairs, int pair_count, int vocab_size) {
    double input_vectors[VOCAB_SIZE][EMBEDDING_SIZE];
    double output_vectors[VOCAB_SIZE][EMBEDDING_SIZE];
    initialize_vectors(input_vectors);
    initialize_vectors(output_vectors);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < pair_count; i++) {
            printf("Epoch %d: softmax %d\n", epoch + 1, i);
            int target = pairs[i].target;
            int context = pairs[i].context;

            double dot_product[EMBEDDING_SIZE];
            for (int k = 0; k < EMBEDDING_SIZE; k++) {
                dot_product[k] = input_vectors[target][k] * output_vectors[context][k];
            }

            double output_prob[VOCAB_SIZE];
            softmax(dot_product, output_prob, VOCAB_SIZE);

            for (int k = 0; k < EMBEDDING_SIZE; k++) {
                double error = (output_prob[context] - 1.0);
                input_vectors[target][k] -= LEARNING_RATE * error * output_vectors[context][k];
                output_vectors[context][k] -= LEARNING_RATE * error * input_vectors[target][k];
            }
        }

        printf("Epoch %d completed. \n", epoch + 1);
    }

    save_vectors(input_vectors, "vector.txt");
}

Pair* generate_pairs(const char* filename, int* pair_count) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("파일 fopen 오류");
        return NULL;
    }

    char line[256];
    int* words = NULL;
    int word_count = 0;

    while (fgets(line, sizeof(line), file)) {
        int index;
        char word[256];
        sscanf(line, "%d %s", &index, word);

        words = (int*)realloc(words, sizeof(int) * (word_count + 1));
        words[word_count] = index;
        word_count++;
    }
    fclose(file);

    Pair* pairs = NULL;
    *pair_count = 0;

    for (int i = 0; i < word_count; i++) {
        for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; j++) {
            if (j != 0 && (i + j) >= 0 && (i + j) < word_count) {
                pairs = (Pair*)realloc(pairs, sizeof(Pair) * (*pair_count + 1));
                pairs[*pair_count].target = words[i];
                pairs[*pair_count].context = words[i + j];
                (*pair_count)++;
            }
        }
    }

    free(words);
    return pairs;
}

void split_sentences(const char* text, FILE* output_file) {
    bool in_quotes = false;
    const char* start = text;
    const char* ptr = text;
    int sentence_num = 0;

    while (*ptr) {
        if (*ptr == '\"') {
            in_quotes = !in_quotes;
        }

        if (!in_quotes && (*ptr == '.' || *ptr == '!' || *ptr == '?')) {
            while (*(ptr + 1) == ' ' || *(ptr + 1) == '\n' || *(ptr + 1) == '\r' || *(ptr + 1) == '\t') {
                ptr++;
            }
            ptr++;

            size_t len = ptr - start;
            if (len > 0) {
                char* sentence = (char*)malloc(len + 1);

                if (sentence == NULL) {
                    perror("메모리 할당 오류");
                    return;
                }

                strncpy(sentence, start, len);
                sentence[len] = '\0';
                fprintf(output_file, "%d %s\n", sentence_num, sentence);
                printf("split_sentences: %d\n", sentence_num);
                free(sentence);
                sentence_num++;
            }

            start = ptr;
        }
        else {
            ptr++;
        }
    }

    if (start != ptr) {
        size_t len = ptr - start;
        if (len > 0) {
            char* sentence = (char*)malloc(len + 1);
            if (sentence == NULL) {
                perror("메모리 할당 오류");
                return;
            }

            strncpy(sentence, start, len);
            sentence[len] = '\0';
            fprintf(output_file, "%d %s\n", sentence_num, sentence);
            printf("split_sentences: %d\n", sentence_num);
            free(sentence);
        }
    }
}


int tokenize() {
    FILE* input_file = fopen("output.txt", "r");
    if (input_file == NULL) {
        perror("파일 fopen 오류");
        return 1;
    }

    fseek(input_file, 0, SEEK_END);
    long file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    char* text = (char*)malloc(file_size + 1);
    if (text == NULL) {
        perror("메모리 malloc 오류");
        fclose(input_file);
        return 1;
    }

    fread(text, 1, file_size, input_file);
    text[file_size] = '\0';

    fclose(input_file);

    FILE* output_file = fopen("tokenized.txt", "w");
    if (output_file == NULL) {
        perror("파일 fopen 오류");
        free(text);
        return 1;
    }

    tokenize_words(text, output_file);

    free(text);
    fclose(output_file);

    return 0;
}

void tokenize_words(const char* text, FILE* output_file) {
    const char* delimiters = " \t\r\n.,!?\"'";
    char* copy = strdup(text);
    if (copy == NULL) {
        perror("메모리 strdup 오류");
        return;
    }

    char* token = strtok(copy, delimiters);
    int word_count = 0;

    while (token != NULL) {
        fprintf(output_file, "%d %s\n", word_count, token);
        printf("tokenize_words: %d\n", word_count);
        word_count++;
        token = strtok(NULL, delimiters);
    }
    free(copy);
}

void save_vectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE], const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("파일 fopen 오류");
        return;
    }

    for (int i = 0; i < VOCAB_SIZE; i++) {
        fprintf(file, "%d ", i);
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            fprintf(file, "%.6f ", vectors[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void compute_tfidf(const char* filename, Word* words, int* word_count) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("파일 fopen 오류");
        return;
    }

    int doc_count = 0;
    char line[256];
    char** docs = NULL;

    // Read documents and store them in memory
    while (fgets(line, sizeof(line), file)) {
        docs = realloc(docs, sizeof(char*) * (doc_count + 1));
        docs[doc_count] = strdup(line);
        doc_count++;
    }
    fclose(file);

    int* term_freqs = calloc(VOCAB_SIZE, sizeof(int));
    int* doc_freqs = calloc(VOCAB_SIZE, sizeof(int));
    int total_terms = 0;

    // Calculate term frequencies
    for (int i = 0; i < doc_count; i++) {
        char* token = strtok(docs[i], " \t\r\n");
        while (token) {
            int index = atoi(token);
            term_freqs[index]++;
            total_terms++;
            token = strtok(NULL, " \t\r\n");
        }
    }

    // Calculate document frequencies
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < doc_count; j++) {
            char* token = strtok(docs[j], " \t\r\n");
            while (token) {
                int index = atoi(token);
                if (index == i) {
                    doc_freqs[i]++;
                    break;
                }
                token = strtok(NULL, " \t\r\n");
            }
        }
    }

    // Calculate TF-IDF values
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (term_freqs[i] > 0) {
            double tf = (double)term_freqs[i] / total_terms;
            double idf = log((double)doc_count / (doc_freqs[i] + 1));
            words[*word_count].tfidf = tf * idf;
            snprintf(words[*word_count].word, 256, "%d", i);
            (*word_count)++;
        }
    }

    free(term_freqs);
    free(doc_freqs);

    for (int i = 0; i < doc_count; i++) {
        free(docs[i]);
    }
    free(docs);
}

void save_tfidf(Word* words, int word_count, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("파일 fopen 오류");
        return;
    }

    for (int i = 0; i < word_count; i++) {
        fprintf(file, "%s %.6f\n", words[i].word, words[i].tfidf);
    }

    fclose(file);
}

Word find_max_tfidf_word(Word* words, int word_count) {
    Word max_word = words[0];
    for (int i = 1; i < word_count; i++) {
        if (words[i].tfidf > max_word.tfidf) {
            max_word = words[i];
        }
    }
    return max_word;
}
