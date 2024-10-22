#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <queue>
#include <omp.h>

using namespace std;

#define EPOCHS 100 // 에포크 수
#define VOCAB_SIZE 650 // 최대 어휘 크기
#define EMBEDDING_SIZE 100 // 단어 임베딩 크기
#define WINDOW_SIZE 10 // Skip-gram 모델에서의 컨텍스트 윈도우 크기
#define LEARNING_RATE 0.5 // 학습률
#define MAX_WORD_LENGTH 100
#define MAX_WORDS 10000
#define MAX_VECTOR_DIMENSION 300 // 최대 벡터 차원

struct Pair {
    int target;
    int context;
};

struct Word {
    string word;
    double tfidf;
    vector<double> vector;
    Word() : vector(EMBEDDING_SIZE) {}
};

struct SentenceVector {
    string sentence;
    vector<double> vector;
    SentenceVector() : vector(EMBEDDING_SIZE) {}
};

struct WordVector {
    string word;
    vector<float> vector;
    WordVector() : vector(MAX_VECTOR_DIMENSION) {}
};

struct TreeNode {
    double value;
    TreeNode* left;
    TreeNode* right;
    int index;
    TreeNode(double v, int i) : value(v), left(nullptr), right(nullptr), index(i) {}
};

// 불용어 목록
const vector<string> stopwords = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", 
    "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", 
    "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", 
    "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", 
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
};

bool isStopword(const string& word) {
    return find(stopwords.begin(), stopwords.end(), word) != stopwords.end();
}

vector<string> splitSentences(const string& text) {
    bool inQuotes = false;
    stringstream sentenceStream;
    vector<string> sentences;
    size_t start = 0;
    
    for (size_t i = 0; i < text.length(); ++i) {
        if (text[i] == '\"') {
            inQuotes = !inQuotes;
        }
        
        if (!inQuotes && (text[i] == '.' || text[i] == '!' || text[i] == '?')) {
            size_t len = i - start + 1;
            if (len > 0) {
                sentences.push_back(text.substr(start, len));
                start = i + 1;
            }
        }
    }
    
    if (start < text.length()) {
        sentences.push_back(text.substr(start));
    }
    
    return sentences;
}

void tokenizeWords(const string& text, vector<string>& tokens) {
    const string delimiters = " \t\r\n.,!?\"'";
    size_t start = text.find_first_not_of(delimiters), end = 0;

    while (start != string::npos) {
        end = text.find_first_of(delimiters, start);
        tokens.push_back(text.substr(start, end - start));
        start = text.find_first_not_of(delimiters, end);
    }
}

Pair* generatePairs(const vector<string>& words, int* pairCount) {
    int wordCount = words.size();
    vector<Pair> pairs;

    for (int i = 0; i < wordCount; i++) {
        for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; j++) {
            if (j != 0 && (i + j) >= 0 && (i + j) < wordCount) {
                Pair pair = {i, i + j};
                pairs.push_back(pair);
            }
        }
    }

    *pairCount = pairs.size();
    return pairs.data();
}
