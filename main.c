#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <Windows.h>
#include <omp.h>

#define EPOCHS 100 // 에포크 수
#define VOCAB_SIZE 600 // 최대 d어휘 크기
#define EMBEDDING_SIZE 100 // 단어 임베딩 크기
#define WINDOW_SIZE 4 // Skip-gram 모델에서의 컨텍스트 윈도우 크기
#define LEARNING_RATE 0.1 // 학습률
#define MAX_WORD_LENGTH 50
#define MAX_WORDS 10000

// 타겟 단어와 컨텍스트 단어 쌍을 나타내느 구조체 ?
typedef struct Pair {
	int target;
	int context;
} Pair;

// 단어와 그 단어의 TF-IDF 값을 나타내는 구조체
typedef struct Word {
	char word[256];
	double tfidf;
	double vector[EMBEDDING_SIZE];
} Word;

// 문장과 문장의 벡터 표현을 나타내는 구조체
typedef struct SentencVector {
	char sentence[1024];
	double vector[EMBEDDING_SIZE];
} SentenceVector;

// tree의 node 구조체
typedef struct TreeNode {
	double value;
	struct TreeNode* left;
	struct TreeNode* right;
	int index;
} TreeNode;

// 불용어 목록
const char* stopwords[] = {
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd",
			"your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
			"herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
			"who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
			"being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
			"or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
			"into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
			"on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
			"how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
			"only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
			"should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
			"couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
			"isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
			"shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
};
const int stopwordsCount = sizeof(stopwords) / sizeof(stopwords[0]);

void initializeVectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE]);
double randomDouble();
void softmax(double* input, double* output, int size);
void train(Pair* pairs, int pairCount, int vocabSize);
Pair* generatePairs(const char* filename, int* pairCount);
void splitSentences(const char* text, FILE* outputFile);
void tokenizeWords(const char* text, FILE* outputFile);
int tokenize();
void saveVectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE], const char* filename);
void computeTfidf(const char* filename, Word* words, int* wordCount);
void saveTfidf(Word* words, int wordCount, const char* filename);
Word findMaxTfidfWord(Word* words, int wordCount);
void printProgressBar(int epoch, int current, int total);
void computeSentenceVectors(const char* filename, Word* words, int wordCount, SentenceVector* sentenceVectors, int* sentenceCount);
void saveSentenceVectors(SentenceVector* sentenceVectors, int sentenceCount, const char* filename);
TreeNode* createNode(double value, int index);
TreeNode* buildTree(double* input, int size);
void calculateProbabilities(TreeNode* node, double probability, double* output);
void freeTree(TreeNode* node);
int isStopword(const char* word);

//continue, else if, 전역변수 <-- 쓰면안됨

int main() {
	// 0. 입력
	printf("Reading dataset.txt");
	FILE* inputFile = fopen("dataset.txt", "r"); // dataset.txt 파일을 읽기 모드로
	if (inputFile == NULL) {
		perror("Error opening dataset.txt"); // 파일 열기 에러메시지
		return 1;
	}
	printf("\rReading dataset.txt completed.\r\n");
	fseek(inputFile, 0, SEEK_END); // 포인터를 파일 끝으로 --> 파일 크기
	long fileSize = ftell(inputFile); // 파일 크기 저장
	fseek(inputFile, 0, SEEK_SET); // 포인터 원위치
	// 메모리 할당 (파일크기로)
	char* text = (char*)malloc(fileSize + 1);
	if (text == NULL) {
		perror("Error allocating memory for dataset text"); //메모리할당 에러메시지
		fclose(inputFile);
		return 1;
	}
	fread(text, 1, fileSize, inputFile); //text = dataset.txt 파일내용
	text[fileSize] = '\0'; // 문자열 끝에 널문자 추가
	fclose(inputFile);

	// 1. 문장 토큰화
	printf("Splitting sentences into sentence_tokenized.txt\n");
	FILE* outputFile = fopen("sentence_tokenized.txt", "w"); // sentence_tokenized.txt 파일을 쓰기 모드로
	if (outputFile == NULL) {
		perror("Error opening sentence_tokenized.txt"); // 파일 열기 에러메시지
		free(text);
		return 1;
	}
	splitSentences(text, outputFile); // 문장분리함수호출
	fclose(outputFile); // 파일닫고
	free(text); // 메모리 할당 해제

	// 2. 단어 토큰화
	printf("Tokenizing sentences into word_tokenized.txt\n");
	if (tokenize() != 0) {
		printf("Tokenization failed.\n"); // 토큰화 에러메시지
		return 1;
	}

	// 3. 정제

	/*
	printf("Removing stopword from word_tokenized.txt\n");
	// 파일 읽기
	FILE* word_file = fopen("word_tokenized.txt", "r");
	if (word_file == NULL) {
		perror("Error opening word_tokenized.txt");
		return 1;
	}

	// wordss에 파일 내용 저장
	char wordss[MAX_WORDS][MAX_WORD_LENGTH];
	int words_count = 0;
	while (fscanf(word_file, "%s", wordss[words_count]) != EOF) {
		words_count++;
	}
	fclose(word_file);

	//파일 쓰기
	FILE* output_word_file = fopen("word_tokenized.txt", "w");
	if (output_word_file == NULL) {
		perror("Error opening word_tokenized.txt for writing");
		return 1;
	}

	//불용어 제거
	int stopwords_removed = 0;
	for (int i = 0; i < words_count; i++) {
		if (!isStopword(wordss[i])) {
			fprintf(output_word_file, "%s\n", wordss[i]);
		}
		else {
			stopwords_removed++;
		}
	}
	fclose(output_word_file);

	printf("\rRemoved %d stopwords \r\n", stopwords_removed);
	*/

	// 4. 쌍연산
	printf("Generating pairs from word_tokenized.txt\n");
	int pairCount;
	Pair* pairs = generatePairs("word_tokenized.txt", &pairCount); // word_tokenized.txt에서 쌍 생성
	if (pairs == NULL) {
		return 1;
	}

	// 5. word2vec 임베딩 (단어)
	for (int i = 0; i < pairCount; i++) {
		printf("\rTraining Skip-Gram model with %d pairs", i);
	}

	train(pairs, pairCount, VOCAB_SIZE); // Skip-Gram 모델 학습
	free(pairs);

	// 6. TF-IDF 중요도 분석기법 -> centeroid 추출
	printf("Computing TF-IDF values\n");
	Word words[VOCAB_SIZE];
	int wordCount = 0;
	computeTfidf("word_tokenized.txt", words, &wordCount); // TF-IDF 값을 계산해서
	saveTfidf(words, wordCount, "weight.txt"); // weight.txt에 저장
	Word maxWord = findMaxTfidfWord(words, wordCount); // TF-IDF 1등 찾기
	printf("Word with highest TF-IDF: %s (%.16f)\n", maxWord.word, maxWord.tfidf);

	// 7. 문장 임베딩
	printf("Computing sentence vectors\n");
	SentenceVector sentenceVectors[100];
	int sentenceCount = 0;
	computeSentenceVectors("sentence_tokenized.txt", words, wordCount, sentenceVectors, &sentenceCount);
	printf("Saving sentence vectors to sentence_vectors.txt\n");
	saveSentenceVectors(sentenceVectors, sentenceCount, "sentence_vectors.txt");

	// 8. centeroid를 바탕으로 문장 cos 유사도 계산

	// 9. 추출, 출력


	return 0;
}

int isStopword(const char* word) {
	for (int i = 0; i < stopwordsCount; i++) {
		if (strcmp(word, stopwords[i]) == 0) {
			return 1;
		}
	}
	return 0;
}


//(+-0.5/EMBEDDING_SIZE 사이의 값으로) 각 단어 벡터를 초기화
void initializeVectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE]) { //vectors <- 초기화할 벡터 배열
	for (int i = 0; i < VOCAB_SIZE; i++) {
		for (int j = 0; j < EMBEDDING_SIZE; j++) {
			vectors[i][j] = (randomDouble() - 0.5) / EMBEDDING_SIZE;
		}
	}
}

//rand()를 이용해 0~RAND_MAX 정수를 생성하고 RAND_MAX로 나눠서 0~1 실수 return
double randomDouble() {
	return (double)rand() / (double)RAND_MAX;
}

/*
기존 softmax 계산 함수
void softmax(double* input, double* output, int size) {
	//overflow 방지를 위해 입력벡터의 최대값 도출
	double max = input[0];
	for (int i = 1; i < size; i++) {
		if (input[i] > max) {
			max = input[i];
		}
	}
	// 지수함수 및 합계 계산 --> 중심적인 softmax 처리과정 (연산량 많음;;)
	double sum = 0.0;
	for (int i = 0; i < size; i++) {
		output[i] = exp(input[i] - max);
		sum += output[i];
	}
	// softmax 처리결가ㅗ를 0과 1 사이의 확률로 나타내지도록 정규화
	for (int i = 0; i < size; i++) {
		output[i] /= sum;
	}
}
*/

//이진 트리를 활용한 softmax 연산량 개선 !!!
void softmax(double* input, double* output, int size) { //input = 입력벡터, output = 출력벡터, size = 벡터크기
	// overflow 방지를 위해 입력벡터의 최대값 도출
	double max = input[0];
	for (int i = 1; i < size; i++) {
		if (input[i] > max) {
			max = input[i];
		}
	}

	// 입력벡터를 트리에 사용할 벡터로 변환
	for (int i = 0; i < size; i++) {
		input[i] = exp(input[i] - max);
	}

	// 이진 트리 구성
	TreeNode* root = buildTree(input, size);

	// 트리를 탐색하며 softmax 확률을 계산
	for (int i = 0; i < size; i++) {
		output[i] = 0.0;
	}
	calculateProbabilities(root, 1.0, output);

	// 트리 메모리 해제
	freeTree(root);
}

// TreeNode 생성
TreeNode* createNode(double value, int index) {
	TreeNode* newNode = (TreeNode*)malloc(sizeof(TreeNode));
	if (newNode != NULL) {
		newNode->value = value;
		newNode->left = NULL;
		newNode->right = NULL;
		newNode->index = index;
		return newNode;
	}
}

// 이진 트리 구성
TreeNode* buildTree(double* input, int size) {
	// 우선순위 큐
	TreeNode** heap = (TreeNode**)malloc(size * sizeof(TreeNode*));

	for (int i = 0; i < size; i++) {
		if (heap != NULL) {
			heap[i] = createNode(input[i], i);
		}
	}

	int heapSize = size;

	// 힙 빌드
	while (heapSize > 1 && heap != NULL) {

		// 최소값 두 개 찾기
		int min1 = 0, min2 = 1;
		if (heap[min2]->value < heap[min1]->value) {
			int temp = min1;
			min1 = min2;
			min2 = temp;
		}
		for (int i = 2; i < heapSize; i++) {
			if (heap[i]->value < heap[min2]->value) {
				if (heap[i]->value < heap[min1]->value) {
					min2 = min1;
					min1 = i;
				}
				else {
					min2 = i;
				}
			}
		}

		// 새로운 부모 노드 생성
		TreeNode* parent = createNode(heap[min1]->value + heap[min2]->value, -1);
		parent->left = heap[min1];
		parent->right = heap[min2];

		// 힙 업데이트
		heap[min1] = parent;
		heap[min2] = heap[heapSize - 1];
		heapSize--;
	}
	if (heap != NULL) {
		TreeNode* root = heap[0];
		free(heap);
		return root;
	}
}

// 트리를 탐색하며 softmax 확률을 계산
void calculateProbabilities(TreeNode* node, double probability, double* output) {
	if (node->left == NULL && node->right == NULL) {
		output[node->index] = probability;
		return;
	}
	if (node->left) {
		calculateProbabilities(node->left, probability * 0.5, output);
	}
	if (node->right) {
		calculateProbabilities(node->right, probability * 0.5, output);
	}
}

// 트리 메모리 해제
void freeTree(TreeNode* node) {
	if (node == NULL) return;
	freeTree(node->left);
	freeTree(node->right);
	free(node);
}

// skip-gram 모델을 학습해서 word_vectors.txt 파일에 저장
void train(Pair* pairs, int pairCount, int vocabSize) { //pairs = 학습할 단어 쌍 배열, pairCount = 단어 쌍의 수, vocabSize =? 크기
	double inputVectors[VOCAB_SIZE][EMBEDDING_SIZE];
	double outputVectors[VOCAB_SIZE][EMBEDDING_SIZE];
	initializeVectors(inputVectors); //입력 벡터 초기화
	initializeVectors(outputVectors); //출려 벡터 초기화

	for (int epoch = 0; epoch < EPOCHS; epoch++) { //에포크 수(하이퍼파라미터) 만큼 연산 반복
#pragma omp parallel for schedule(dynamic) //병렬 루프 지정 + 동적(dynamic) 작업 할당
		for (int i = 0; i < pairCount; i++) {
			// 타겟 단어와 문맥 단어(window로 잡은 단어)의 내적 계산
			int target = pairs[i].target;
			int context = pairs[i].context;
			double dotProduct[EMBEDDING_SIZE];
			for (int k = 0; k < EMBEDDING_SIZE; k++) {
				dotProduct[k] = inputVectors[target][k] * outputVectors[context][k];
			}

			//내적 계산 결과를 softmax계산 함수에 넣어가꼬 확률 분포 도출
			double outputProb[VOCAB_SIZE];
			softmax(dotProduct, outputProb, VOCAB_SIZE);

			for (int k = 0; k < EMBEDDING_SIZE; k++) {
				double error = (outputProb[context] - 1.0); //예측된 확률과 실제 값을 비교 --> 오차 계산
				//그래디언트 역전파를 이용해 입출력 벡터 업데이트 (얘도 연산량 ㅈ됨;;)
#pragma omp atomic
//특정 메모리 위치에 대한 원자적 연산(뭔말인지모름) 수행 --> 여러 스레드의 동시 접근으로 인한 충돌 방지
				inputVectors[target][k] -= LEARNING_RATE * error * outputVectors[context][k];
#pragma omp atomic
				outputVectors[context][k] -= LEARNING_RATE * error * inputVectors[target][k];
			}

			// 프로그레스바 출력
#pragma omp single //<-- 단일 스레드에서 처리
			printProgressBar(epoch + 1, i + 1, pairCount);
		}
		printf("Epoch %d: completed.\n", epoch + 1);
	}
	saveVectors(inputVectors, "word_vectors.txt"); //학습된 벡터 저장
}

//텍스트 파일을 읽어서 단어 쌍 생성, 생성된 쌍을 배열로 반환
Pair* generatePairs(const char* filename, int* pairCount) { //filename = 읽을 텍스트 파일명, pairCount = 생성된 쌍의 수
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening tokenized.txt");
		return NULL;
	}

	char line[256];
	int* words = NULL;
	int wordCount = 0;
	size_t wordsAllocSize = 1024;

	words = (int*)malloc(sizeof(int) * wordsAllocSize);
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

		if (wordCount >= wordsAllocSize) {
			wordsAllocSize *= 2;

			if (words == NULL) {
				perror("Error reallocating memory for words");
				fclose(file);
				return NULL;
			}
			else {

				words = (int*)realloc(words, sizeof(int) * wordsAllocSize);
			}
		}
		words[wordCount] = index;
		wordCount++;
	}
	fclose(file);

	Pair* pairs = NULL;
	size_t pairsAllocSize = 1024;
	*pairCount = 0;

	pairs = (Pair*)malloc(sizeof(Pair) * pairsAllocSize);
	if (pairs == NULL) {
		perror("Error allocating memory for pairs");
		free(words);
		return NULL;
	}

	for (int i = 0; i < wordCount; i++) {
		for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; j++) {
			if (j != 0 && (i + j) >= 0 && (i + j) < wordCount) {
				if (*pairCount >= pairsAllocSize) {
					pairsAllocSize *= 2;
					pairs = (Pair*)realloc(pairs, sizeof(Pair) * pairsAllocSize);
					if (pairs == NULL) {
						perror("Error reallocating memory for pairs");
						free(words);
						return NULL;
					}
				}
				pairs[*pairCount].target = words[i];
				pairs[*pairCount].context = words[i + j];
				(*pairCount)++;
			}
		}
	}
	free(words);
	return pairs;
}

//문장 단위로 토큰화
void splitSentences(const char* text, FILE* outputFile) {
	bool inQuotes = false;
	const char* start = text;
	const char* ptr = text;
	int sentenceNum = 0;

	while (*ptr) {
		if (*ptr == '\"') {
			inQuotes = !inQuotes;
		}

		if (!inQuotes && (*ptr == '.' || *ptr == '!' || *ptr == '?')) {
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
				fprintf(outputFile, "%d %s\n", sentenceNum, sentence);
				free(sentence);
				sentenceNum++;
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
			fprintf(outputFile, "%d %s\n", sentenceNum, sentence);
			free(sentence);
		}
	}
}

//sentence_tokenized.txt 를 읽어서 각 문장을 단어로 분리하여 word_tokenized.txt에 저장
int tokenize() {
	FILE* inputFile = fopen("sentence_tokenized.txt", "r");
	if (inputFile == NULL) {
		perror("Error opening sentence_tokenized.txt");
		return 1;
	}

	fseek(inputFile, 0, SEEK_END);
	long fileSize = ftell(inputFile);
	fseek(inputFile, 0, SEEK_SET);

	char* text = (char*)malloc(fileSize + 1);
	if (text == NULL) {
		perror("Error allocating memory for tokenization");
		fclose(inputFile);
		return 1;
	}

	fread(text, 1, fileSize, inputFile);
	text[fileSize] = '\0';

	fclose(inputFile);

	FILE* outputFile = fopen("word_tokenized.txt", "w");
	if (outputFile == NULL) {
		perror("Error opening word_tokenized.txt");
		free(text);
		return 1;
	}

	tokenizeWords(text, outputFile);

	free(text);
	fclose(outputFile);

	return 0;
}

//실질적힌 단어 토큰화 처리 수행
void tokenizeWords(const char* text, FILE* outputFile) {
	const char* delimiters = " \t\r\n.,!?\"'";
	char* copy = strdup(text);
	if (copy == NULL) {
		perror("Error duplicating text for tokenization");
		return;
	}

	char* token = strtok(copy, delimiters);
	int wordCount = 0;
	printf("");
	while (token != NULL) {
		fprintf(outputFile, "%d %s\n", wordCount, token);
		printf("\rtokenizeWords: %d", wordCount);
		wordCount++;
		token = strtok(NULL, delimiters);
	}
	printf("\n");
	free(copy);
}


//벡터를 파일에 저장
void saveVectors(double vectors[VOCAB_SIZE][EMBEDDING_SIZE], const char* filename) { //vectors = 저장할 벡터 배열
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error opening word_vectors.txt");
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

//단어의 tfidf 값을 연산
void computeTfidf(const char* filename, Word* words, int* wordCount) { //filename = 텍스트파일이름, words = 단어 배열
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening word_tokenized.txt for TF-IDF computation");
		return;
	}

	int docCount = 0;
	char line[256];
	char** docs = NULL;

	// 문서를 읽어서 메모리에 저장
	while (fgets(line, sizeof(line), file)) {
		docs = realloc(docs, sizeof(char*) * (docCount + 1));
		if (docs == NULL) {
			perror("Error reallocating memory for documents");
			fclose(file);
			return;
		}
		docs[docCount] = strdup(line);
		if (docs[docCount] == NULL) {
			perror("Error duplicating line for document storage");
			fclose(file);
			return;
		}
		docCount++;
	}
	fclose(file);

	int* termFreqs = calloc(VOCAB_SIZE, sizeof(int));
	int* docFreqs = calloc(VOCAB_SIZE, sizeof(int));
	if (termFreqs == NULL || docFreqs == NULL) {
		perror("Error allocating memory for term/document frequencies");
		free(termFreqs);
		free(docFreqs);
		for (int i = 0; i < docCount; i++) {
			free(docs[i]);
		}
		free(docs);
		return;
	}
	int totalTerms = 0;

	// Calculate term frequencies
	for (int i = 0; i < docCount; i++) {
		char* token = strtok(docs[i], " \t\r\n");
		while (token) {
			int index = atoi(token);
			termFreqs[index]++;
			totalTerms++;
			token = strtok(NULL, " \t\r\n");
		}
	}

	// Calculate document frequencies
	for (int i = 0; i < VOCAB_SIZE; i++) {
		for (int j = 0; j < docCount; j++) {
			char* token = strtok(docs[j], " \t\r\n");
			while (token) {
				int index = atoi(token);
				if (index == i) {
					docFreqs[i]++;
					break;
				}
				token = strtok(NULL, " \t\r\n");
			}
		}
	}

	// Calculate TF-IDF values
	for (int i = 0; i < VOCAB_SIZE; i++) {
		if (termFreqs[i] > 0) {
			double tf = (double)termFreqs[i] / totalTerms;
			double idf = log((double)docCount / (docFreqs[i] + 1));
			words[*wordCount].tfidf = tf * idf;
			snprintf(words[*wordCount].word, 256, "%d", i);
			(*wordCount)++;
		}
	}
	free(termFreqs);
	free(docFreqs);

	for (int i = 0; i < docCount; i++) {
		free(docs[i]);
	}
	free(docs);
}

//계산된 tfidf 값을 파일로 저장
void saveTfidf(Word* words, int wordCount, const char* filename) { //상동
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error opening weight.txt");
		return;
	}

	for (int i = 0; i < wordCount; i++) {
		fprintf(file, "%s %.16f\n", words[i].word, words[i].tfidf);
	}

	fclose(file);
}

//tfidf 값이 가장 높은 단어 탐색
Word findMaxTfidfWord(Word* words, int wordCount) {
	Word maxWord = words[0];
	for (int i = 1; i < wordCount; i++) {
		if (words[i].tfidf > maxWord.tfidf) {
			maxWord = words[i];
		}
	}
	return maxWord;
}

//프로그레스바 출력
void printProgressBar(int epoch, int current, int total) {
	int barWidth = 50;
	float progress = (float)current / total;
	int pos = (int)(barWidth * progress);

	printf("Epoch %d: |", epoch);
	for (int i = 0; i < barWidth; ++i) {
		if (i <= pos) {
			printf("▒");
		}
		else {
			printf(" ");
		}
	}
	printf("| %d% %(%d/%d)\r", (int)(progress * 100), current, total);
	fflush(stdout);

	if (current == total) {
		printf("Epoch %d: completed.                                                          \r", epoch);
	}
}

/*
void computeSentenceVectors(const char* filename, Word* words, int wordCount, SentenceVector* sentenceVectors, int* sentenceCount) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening sentence_tokenized.txt");
		return;
	}
	char line[1024];
	*sentenceCount = 0;
	while (fgets(line, sizeof(line), file)) {
		// 문장을 복사하여 버퍼 오버플로우 방지
		strncpy(sentenceVectors[*sentenceCount].sentence, line, sizeof(sentenceVectors[*sentenceCount].sentence));
		double vector_sum[EMBEDDING_SIZE] = { 0.0 }; // 벡터 합 초기화
		int word_index;
		char* token = strtok(line, " \n");
		int word_count_in_sentence = 0;

		while (token != NULL) {
			//읽어온 인덱스가 유효한 범위인지 확인
			if (sscanf(token, "%d", &word_index) == 1 && word_index >= 0 && word_index < word_count) {
				for (int i = 0; i < EMBEDDING_SIZE; i++) {
					vector_sum[i] += words[word_index].tfidf * words[word_index].vector[i];
				}
				word_count_in_sentence++;
			}
			token = strtok(NULL, " \n");
		}
		// 문장에서 단어가 하나 이상일 때만 평균 계산
		if (word_count_in_sentence > 0) {
			for (int i = 0; i < EMBEDDING_SIZE; i++) {
				sentenceVectors[*sentenceCount].vector[i] = vector_sum[i] / word_count_in_sentence; // 벡터의 평균 계산
			}
		} else {
			// 문장이 비어있거나 유효한 단어 인덱스가 없는 경우 0 벡터로 유지
			memset(sentenceVectors[*sentenceCount].vector, 0, sizeof(sentenceVectors[*sentenceCount].vector));
		}
		(*sentenceCount)++;
	}
	fclose(file);
}
*/

//각 문장의 벡터를 계산하여 sentenceVectors 배열에 저장
void computeSentenceVectors(const char* filename, Word* words, int wordCount, SentenceVector* sentenceVectors, int* sentenceCount) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening sentence_tokenized.txt");
		return;
	}

	char line[1024];
	int sentenceIdx = 0;

	while (fgets(line, sizeof(line), file)) {
		SentenceVector sv;
		strcpy(sv.sentence, line);
		memset(sv.vector, 0, sizeof(sv.vector));

		char* token = strtok(line, " \t\r\n");
		while (token != NULL) {
			for (int i = 0; i < wordCount; i++) {
				if (strcmp(token, words[i].word) == 0) {
					for (int j = 0; j < EMBEDDING_SIZE; j++) {
						sv.vector[j] += words[i].vector[j];
					}
					break;
				}
			}
			token = strtok(NULL, " \t\r\n");
		}
		sentenceVectors[sentenceIdx++] = sv;
	}
	*sentenceCount = sentenceIdx;
	fclose(file);
}

//문장 벡터를 파일에 저장
void saveSentenceVectors(SentenceVector* sentenceVectors, int sentenceCount, const char* filename) {
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error opening sentence_vectors.txt");
		return;
	}

	for (int i = 0; i < sentenceCount; i++) {
		fprintf(file, "%s", sentenceVectors[i].sentence);
		for (int j = 0; j < EMBEDDING_SIZE; j++) {
			fprintf(file, "%.20f ", sentenceVectors[i].vector[j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}

