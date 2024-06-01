#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <Windows.h>
#include <omp.h>

#define VOCAB_SIZE 1000 // 최대 어휘 크기 ?
#define EMBEDDING_SIZE 100 // 단어 임베딩 크기
#define WINDOW_SIZE 2 // Skip-gram 모델에서의 컨텍스트 윈도우 크기
#define LEARNING_RATE 0.1 // 학습률
#define EPOCHS 50 // 에포크 수

// 타겟 단어와 컨텍스트 단어 쌍을 나타내느 구조체 ?
typedef struct {
	int target;
	int context;
} Pair;

// 단어와 그 단어의 TF-IDF 값을 나타내는 구조체
typedef struct {
	char word[256];
	double tfidf;
} Word;

// 문장과 그 그 문장의 벡터 표현을 나타내는 구조체
typedef struct {
	char sentence[1024];
	double vector[EMBEDDING_SIZE];
} SentenceVector;

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
void print_progress_bar(int epoch, int current, int total);
void compute_sentence_vectors(const char* filename, Word* words, int word_count, SentenceVector* sentence_vectors, int* sentence_count);
void save_sentence_vectors(SentenceVector* sentence_vectors, int sentence_count, const char* filename);


int main() {
	// 0. 입력값
	printf("Reading input.txt\n");
	FILE* input_file = fopen("input.txt", "r"); // input.txt 파일을 읽기 모드로
	if (input_file == NULL) {
		perror("Error opening input.txt"); // 파일 열기 에러메시지
		return 1;
	}
	printf("Reading input.txt complete\n");
	fseek(input_file, 0, SEEK_END); // 포인터를 파일 끝으로 --> 파일 크기
	long file_size = ftell(input_file); // 파일 크기 저장
	fseek(input_file, 0, SEEK_SET); // 포인터 원위치
	// 메모리 할당 (파일크기로)
	char* text = (char*)malloc(file_size + 1);
	if (text == NULL) {
		perror("Error allocating memory for input text"); //메모리할당 에러메시지
		fclose(input_file);
		return 1;
	}
	fread(text, 1, file_size, input_file); //text = input.txt 파일내용
	text[file_size] = '\0'; // 문자열 끝에 널문자 추가
	fclose(input_file);

	/*1. 문장 토큰화*/
	printf("Splitting sentences into sentence_tokenized.txt\n");
	FILE* output_file = fopen("sentence_tokenized.txt", "w"); // sentence_tokenized.txt 파일을 쓰기 모드로
	if (output_file == NULL) {
		perror("Error opening sentence_tokenized.txt"); // 파일 열기 에러메시지
		free(text);
		return 1;
	}
	split_sentences(text, output_file); // 문장분리함수호출
	fclose(output_file); // 파일닫고
	free(text); // 메모리 할당 해제

	/*2. 단어 토큰화*/
	printf("Tokenizing sentences into tokenized.txt\n");
	if (tokenize() != 0) {
		printf("Tokenization failed.\n"); // 토큰화 에러메시지
		return 1;
	}
	/*2-1. 쌍연산*/
	printf("Generating pairs from tokenized.txt\n");
	int pair_count;
	Pair* pairs = generate_pairs("tokenized.txt", &pair_count); // tokenized.txt에서 쌍 생성
	if (pairs == NULL) {
		return 1;
	}

	printf("Training Skip-Gram model with %d pairs\n", pair_count);

	train(pairs, pair_count, VOCAB_SIZE); // Skip-Gram 모델 학습
	free(pairs);

	printf("Computing TF-IDF values\n");

	Word words[VOCAB_SIZE];
	int word_count = 0;
	compute_tfidf("tokenized.txt", words, &word_count); // TF-IDF 값 계산합니다.
	save_tfidf(words, word_count, "weight.txt"); // TF-IDF 값 weight.txt에 저장

	Word max_word = find_max_tfidf_word(words, word_count); // TF-IDF 1등 찾기
	printf("Word with highest TF-IDF: %s (%.10f)\n", max_word.word, max_word.tfidf);

	/*3. 문장 벡터 계산*/
	printf("Computing sentence vectors\n");
	SentenceVector sentence_vectors[1000];
	int sentence_count = 0;
	compute_sentence_vectors("sentence_tokenized.txt", words, word_count, sentence_vectors, &sentence_count);

	/*4. 문장 벡터 저장*/
	printf("Saving sentence vectors to sentence_vectors.txt\n");
	save_sentence_vectors(sentence_vectors, sentence_count, "sentence_vectors.txt");

	return 0;
}


/*
	(+-0.5/EMBEDDING_SIZE 사이의 값으로) 각 단어 벡터를 랜덤하게 초기화
*/
void initialize_vectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE]) { //vectors <- 초기화할 벡터 배열
	for (int i = 0; i < VOCAB_SIZE; i++) {
		for (int j = 0; j < EMBEDDING_SIZE; j++) {
			vectors[i][j] = (random_double() - 0.5) / EMBEDDING_SIZE;
		}
	}
}

/*
	rand()를 이용해 0~RAND_MAX 정수 생성
	이거를 RAND_MAX로 나눠서 0~1 실수 return
*/
double random_double() {
	return (double)rand() / (double)RAND_MAX;
}

/*
	input에 대해 소프트맥스 함수로 확률 분포를 계산해서 output에 저장
*/
void softmax(double* input, double* output, int size) { //input = 입력벡터, output = 출력벡터, size = 벡터크기
	double max = input[0];
	for (int i = 1; i < size; i++) {
		if (input[i] > max) {
			max = input[i];
		}
	}

	double sum = 0.0;
	for (int i = 0; i < size; i++) {
		output[i] = exp(input[i] - max); //softmax 계산
		sum += output[i];
	}

	for (int i = 0; i < size; i++) {
		output[i] /= sum; //softmax 계산 결과 정규화
	}
}

/*
	skip-gram 모델을 학습해서 vector.txt 파일에 저장
*/
void train(Pair* pairs, int pair_count, int vocab_size) { //pairs = 학습할 단어 쌍 배열, pair_count = 단어 쌍의 수, vocab_size =? 크기
	double input_vectors[VOCAB_SIZE][EMBEDDING_SIZE];
	double output_vectors[VOCAB_SIZE][EMBEDDING_SIZE];
	initialize_vectors(input_vectors); //입력 벡터 초기화
	initialize_vectors(output_vectors); //출려 ㄱ벡터 초기화

	for (int epoch = 0; epoch < EPOCHS; epoch++) {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < pair_count; i++) {
			int target = pairs[i].target;
			int context = pairs[i].context;

			double dot_product[EMBEDDING_SIZE];
			for (int k = 0; k < EMBEDDING_SIZE; k++) {
				dot_product[k] = input_vectors[target][k] * output_vectors[context][k];
			}

			double output_prob[VOCAB_SIZE];
			softmax(dot_product, output_prob, VOCAB_SIZE); //softmax 계산 함수

			for (int k = 0; k < EMBEDDING_SIZE; k++) {
				double error = (output_prob[context] - 1.0); //손실
#pragma omp atomic
				input_vectors[target][k] -= LEARNING_RATE * error * output_vectors[context][k];
#pragma omp atomic
				output_vectors[context][k] -= LEARNING_RATE * error * input_vectors[target][k];
			}

			// 진행 바 출력
#pragma omp single
			print_progress_bar(epoch + 1, i + 1, pair_count); //진행바 출력
		}
		printf("Epoch %d: completed.\n", epoch + 1);
	}

	save_vectors(input_vectors, "vector.txt"); //학습된 벡터 저장
}

/*
	텍스트 파일을 읽어서 단어 쌍 생성, 생성된 쌍을 배열로 반환
*/
Pair* generate_pairs(const char* filename, int* pair_count) { //filename = 읽을 텍스트 파일명, pair_count = 생성된 쌍의 수
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening tokenized.txt");
		return NULL;
	}

	char line[256];
	int* words = NULL;
	int word_count = 0;
	size_t words_alloc_size = 1024;

	words = (int*)malloc(sizeof(int) * words_alloc_size);
	if (words == NULL) {
		perror("Error allocating memory for words");
		fclose(file);
		return NULL;
	}

	while (fgets(line, sizeof(line), file)) {
		int index;
		char word[256];
		if (sscanf(line, "%d %s", &index, word) != 2) {
			fprintf(stderr, "Error parsing line: %s\n", line);
			free(words);
			fclose(file);
			return NULL;
		}

		if (word_count >= words_alloc_size) {
			words_alloc_size *= 2;
			words = (int*)realloc(words, sizeof(int) * words_alloc_size);
			if (words == NULL) {
				perror("Error reallocating memory for words");
				fclose(file);
				return NULL;
			}
		}
		words[word_count] = index;
		word_count++;
	}
	fclose(file);

	Pair* pairs = NULL;
	size_t pairs_alloc_size = 1024;
	*pair_count = 0;

	pairs = (Pair*)malloc(sizeof(Pair) * pairs_alloc_size);
	if (pairs == NULL) {
		perror("Error allocating memory for pairs");
		free(words);
		return NULL;
	}

	for (int i = 0; i < word_count; i++) {
		for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; j++) {
			if (j != 0 && (i + j) >= 0 && (i + j) < word_count) {
				if (*pair_count >= pairs_alloc_size) {
					pairs_alloc_size *= 2;
					pairs = (Pair*)realloc(pairs, sizeof(Pair) * pairs_alloc_size);
					if (pairs == NULL) {
						perror("Error reallocating memory for pairs");
						free(words);
						return NULL;
					}
				}
				pairs[*pair_count].target = words[i];
				pairs[*pair_count].context = words[i + j];
				(*pair_count)++;
			}
		}
	}

	free(words);
	return pairs;
}


/*
	문장 단위로 토큰화
*/
void split_sentences(const char* text, FILE* output_file) { //text = 입력 텍스트, output_file = 출력 파일
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
					perror("Error allocating memory for sentence");
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
				perror("Error allocating memory for sentence");
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

/*
	sentence_tokenized.txt 를 읽어서 각 문장을 단어로 분리하여 tokenized.txt에 저장
*/
int tokenize() {
	FILE* input_file = fopen("sentence_tokenized.txt", "r");
	if (input_file == NULL) {
		perror("Error opening sentence_tokenized.txt");
		return 1;
	}

	fseek(input_file, 0, SEEK_END);
	long file_size = ftell(input_file);
	fseek(input_file, 0, SEEK_SET);

	char* text = (char*)malloc(file_size + 1);
	if (text == NULL) {
		perror("Error allocating memory for tokenization");
		fclose(input_file);
		return 1;
	}

	fread(text, 1, file_size, input_file);
	text[file_size] = '\0';

	fclose(input_file);

	FILE* output_file = fopen("tokenized.txt", "w");
	if (output_file == NULL) {
		perror("Error opening tokenized.txt");
		free(text);
		return 1;
	}

	tokenize_words(text, output_file);

	free(text);
	fclose(output_file);

	return 0;
}

/*
	실질적힌 단어 토큰화 처리 수행
*/
void tokenize_words(const char* text, FILE* output_file) {
	const char* delimiters = " \t\r\n.,!?\"'";
	char* copy = strdup(text);
	if (copy == NULL) {
		perror("Error duplicating text for tokenization");
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

/*
	벡터를 파일에 저장
*/
void save_vectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE], const char* filename) { //vectors = 저장할 벡터 배열
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error opening vector.txt");
		return;
	}

	for (int i = 0; i < VOCAB_SIZE; i++) {
		fprintf(file, "%d ", i);
		for (int j = 0; j < EMBEDDING_SIZE; j++) {
			fprintf(file, "%.20f ", vectors[i][j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

/*
	단어의 tfidf 값을 연산
*/
void compute_tfidf(const char* filename, Word* words, int* word_count) { //filename = 텍스트파일이름, words = 단어 배열
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening tokenized.txt for TF-IDF computation");
		return;
	}

	int doc_count = 0;
	char line[256];
	char** docs = NULL;

	// Read documents and store them in memory
	while (fgets(line, sizeof(line), file)) {
		docs = realloc(docs, sizeof(char*) * (doc_count + 1));
		if (docs == NULL) {
			perror("Error reallocating memory for documents");
			fclose(file);
			return;
		}
		docs[doc_count] = strdup(line);
		if (docs[doc_count] == NULL) {
			perror("Error duplicating line for document storage");
			fclose(file);
			return;
		}
		doc_count++;
	}
	fclose(file);

	int* term_freqs = calloc(VOCAB_SIZE, sizeof(int));
	int* doc_freqs = calloc(VOCAB_SIZE, sizeof(int));
	if (term_freqs == NULL || doc_freqs == NULL) {
		perror("Error allocating memory for term/document frequencies");
		free(term_freqs);
		free(doc_freqs);
		for (int i = 0; i < doc_count; i++) {
			free(docs[i]);
		}
		free(docs);
		return;
	}
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


/*
	계산된 tfidf 값을 파일로 저장
*/
void save_tfidf(Word* words, int word_count, const char* filename) { //상동
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error opening weight.txt");
		return;
	}

	for (int i = 0; i < word_count; i++) {
		fprintf(file, "%s %.10f\n", words[i].word, words[i].tfidf);
	}

	fclose(file);
}

/*
	tfidf 값이 가장 높은 단어 탐색
*/
Word find_max_tfidf_word(Word* words, int word_count) {
	Word max_word = words[0];
	for (int i = 1; i < word_count; i++) {
		if (words[i].tfidf > max_word.tfidf) {
			max_word = words[i];
		}
	}
	return max_word;
}

/*
	프로그레스바 출력
*/
void print_progress_bar(int epoch, int current, int total) {
	int bar_width = 50;
	float progress = (float)current / total;
	int pos = (int)(bar_width * progress);

	printf("Epoch %d: [", epoch);
	for (int i = 0; i < bar_width; ++i) {
		if (i < pos) {
			printf("=");
		}
		else if (i == pos) {
			printf(">");
		}
		else {
			printf(" ");
		}
	}
	printf("] %d%%(%d/%d)\r", (int)(progress * 100), current, total);
	fflush(stdout);

	if (current == total) {
		printf("Epoch %d: completed.                                                          \r", epoch);
	}
}

/*각 문장의 벡터를 계산하여 sentence_vectors 배열에 저장*/
void compute_sentence_vectors(const char* filename, Word* words, int word_count, SentenceVector* sentence_vectors, int* sentence_count) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening sentence_tokenized.txt");
		return;
	}

	char line[1024];
	*sentence_count = 0;
	while (fgets(line, sizeof(line), file)) {
		strncpy(sentence_vectors[*sentence_count].sentence, line, sizeof(line));

		double vector_sum[EMBEDDING_SIZE] = { 0.0 };
		int word_index;
		char word[256];
		char* token = strtok(line, " \n");
		int word_count_in_sentence = 0;
		while (token != NULL) {
			sscanf(token, "%d", &word_index);
			for (int i = 0; i < EMBEDDING_SIZE; i++) {
				vector_sum[i] += words[word_index].tfidf * vector_sum[i];
			}
			word_count_in_sentence++;
			token = strtok(NULL, " \n");
		}
		for (int i = 0; i < EMBEDDING_SIZE; i++) {
			sentence_vectors[*sentence_count].vector[i] = vector_sum[i] / word_count_in_sentence;
		}
		(*sentence_count)++;
	}
	fclose(file);
}

/*
	문장 벡터를 파일에 저장
*/
void save_sentence_vectors(SentenceVector* sentence_vectors, int sentence_count, const char* filename) {
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error opening sentence_vectors.txt");
		return;
	}

	for (int i = 0; i < sentence_count; i++) {
		fprintf(file, "%s", sentence_vectors[i].sentence);
		for (int j = 0; j < EMBEDDING_SIZE; j++) {
			fprintf(file, "%.10f ", sentence_vectors[i].vector[j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}
