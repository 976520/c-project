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

// 벡터를 파일에 저장
void saveVectors(const vector<vector<double>>& vectors, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    for (int i = 0; i < VOCAB_SIZE; i++) {
        file << i << " ";
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            file << vectors[i][j] << " ";
        }
        file << "\n";
    }
    file.close();
}

// Softmax 계산
void softmax(vector<double>& input, vector<double>& output) {
    double max = *max_element(input.begin(), input.end());
    double sum = 0.0;
    
    for (double& value : input) {
        value = exp(value - max);
        sum += value;
    }
    
    for (double& value : output) {
        value /= sum;
    }
}

// 벡터 초기화
void initializeVectors(vector<vector<double>>& vectors) {
    for (auto& vec : vectors) {
        generate(vec.begin(), vec.end(), []() { return ((double)rand() / RAND_MAX - 0.5) / EMBEDDING_SIZE; });
    }
}

// 랜덤 실수 생성
double randomDouble() {
    return (double)rand() / RAND_MAX;
}

// 트리 노드 생성
struct TreeNode {
    double value;
    int index;
    TreeNode* left;
    TreeNode* right;

    TreeNode(double val, int idx) : value(val), index(idx), left(nullptr), right(nullptr) {}
};

// 이진 트리 구성
TreeNode* buildTree(const vector<double>& input) {
    priority_queue<TreeNode*, vector<TreeNode*>, function<bool(TreeNode*, TreeNode*)>> pq([](TreeNode* a, TreeNode* b) {
        return a->value > b->value;
    });

    for (int i = 0; i < input.size(); ++i) {
        pq.push(new TreeNode(input[i], i));
    }

    while (pq.size() > 1) {
        TreeNode* left = pq.top(); pq.pop();
        TreeNode* right = pq.top(); pq.pop();

        TreeNode* parent = new TreeNode(left->value + right->value, -1);
        parent->left = left;
        parent->right = right;

        pq.push(parent);
    }

    return pq.top();
}

// 트리를 순회하며 softmax 확률 계산
void calculateProbabilities(TreeNode* node, double probability, vector<double>& output) {
    if (!node->left && !node->right) {
        output[node->index] = probability;
        return;
    }
    if (node->left) calculateProbabilities(node->left, probability * 0.5, output);
    if (node->right) calculateProbabilities(node->right, probability * 0.5, output);
}

// 트리 메모리 해제
void freeTree(TreeNode* node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

// 프로그레스 바 출력
void printProgressBar(int epoch, int current, int total) {
    int barWidth = 50;
    float progress = (float)current / total;
    int pos = (int)(barWidth * progress);

    cout << "Epoch " << epoch << ": |";
    for (int i = 0; i < barWidth; ++i) {
        if (i <= pos) cout << "▒";
        else cout << " ";
    }
    cout << "| " << (int)(progress * 100.0) << "% (" << current << "/" << total << ")\r";
    cout.flush();

    if (current == total) {
        cout << "Epoch " << epoch << ": completed." << string(50, ' ') << "\r" << endl;
    }
}

// TF-IDF 계산
void computeTfidf(const string& filename, vector<Word>& words, int& wordCount) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    vector<string> docs;
    string line;
    while (getline(file, line)) {
        docs.push_back(line);
    }

    vector<int> termFreqs(VOCAB_SIZE, 0), docFreqs(VOCAB_SIZE, 0);
    int totalTerms = 0;
    for (const string& doc : docs) {
        stringstream ss(doc);
        string token;
        while (ss >> token) {
            int index = stoi(token);
            termFreqs[index]++;
            totalTerms++;
        }
    }

    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (const string& doc : docs) {
            stringstream ss(doc);
            string token;
            while (ss >> token) {
                if (stoi(token) == i) {
                    docFreqs[i]++;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (termFreqs[i] > 0) {
            double tf = (double)termFreqs[i] / totalTerms;
            double idf = log((double)docs.size() / (docFreqs[i] + 1));
            words[wordCount].tfidf = tf * idf;
            words[wordCount].word = to_string(i);
            wordCount++;
        }
    }
}

// TF-IDF 값을 파일에 저장
void saveTfidf(const vector<Word>& words, int wordCount, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    for (int i = 0; i < wordCount; i++) {
        file << words[i].word << " " << words[i].tfidf << endl;
    }

    file.close();
}

// TF-IDF 값이 가장 높은 단어를 찾는 함수
Word findMaxTfidfWord(const vector<Word>& words, int wordCount) {
    return *max_element(words.begin(), words.begin() + wordCount, [](const Word& a, const Word& b) {
        return a.tfidf < b.tfidf;
    });
}

// 문장 벡터 계산
void computeSentenceVectors(const string& filename, const vector<Word>& words, int wordCount, vector<SentenceVector>& sentenceVectors, int& sentenceCount) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        SentenceVector sv;
        sv.sentence = line;

        stringstream ss(line);
        string token;
        while (ss >> token) {
            for (const Word& word : words) {
                if (word.word == token) {
                    for (int j = 0; j < EMBEDDING_SIZE; j++) {
                        sv.vector[j] += word.vector[j];
                    }
                    break;
                }
            }
        }
        sentenceVectors[sentenceCount++] = sv;
    }

    file.close();
}

// 문장 벡터를 파일에 저장
void saveSentenceVectors(const vector<SentenceVector>& sentenceVectors, int sentenceCount, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    for (int i = 0; i < sentenceCount; i++) {
        file << sentenceVectors[i].sentence;
        for (double value : sentenceVectors[i].vector) {
            file << " " << value;
        }
        file << "\n";
    }

    file.close();
}

// 유클리드 거리 계산
double euclideanDistance(const vector<float>& v1, const vector<float>& v2) {
    double sum = 0.0;
    for (int i = 0; i < v1.size(); i++) {
        sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
}

// 코사인 유사도 계산
double cosineSimilarity(const vector<double>& v1, const vector<double>& v2) {
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;

    for (int i = 0; i < v1.size(); i++) {
        dotProduct += v1[i] * v2[i];
        normA += v1[i] * v1[i];
        normB += v2[i] * v2[i];
    }

    if (normA == 0 || normB == 0) return 0.0;
    return dotProduct / (sqrt(normA) * sqrt(normB));
}

// 가장 가까운 이웃 찾기
void findNearestNeighbor(const string& word, const vector<WordVector>& wordVectors, int numWords) {
    float minDistance = INFINITY;
    string nearestNeighbor;

    for (int i = 0; i < numWords; i++) {
        if (wordVectors[i].word != word) {
            float distance = euclideanDistance(wordVectors[i].vector, wordVectors[i].vector);
            if (distance < minDistance) {
                minDistance = distance;
                nearestNeighbor = wordVectors[i].word;
            }
        }
    }

    cout << word << "'s nearest neighbor: " << nearestNeighbor << endl;
}

