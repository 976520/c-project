import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

def summarize(text, num_sentences=2):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    # 단어 빈도 계산
    freq_table = defaultdict(int)
    for word in words:
        if word not in stop_words:
            freq_table[word] += 1

    # 문장 점수 계산
    sentence_scores = defaultdict(int)
    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                sentence_scores[sentence] += freq_table[word]

    # 점수가 높은 문장 return
    return ' '.join(sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences])


text = """
On the afternoon of the 23rd, at around 4:49 PM, a fire broke out on the 13th floor of an apartment building in Ssangmun-dong, Dobong-gu, Seoul. 
The fire caused significant alarm among residents, and a total of seven people evacuated from the scene. 
Fortunately, thanks to the swift evacuation, there were no casualties. 
Although the fire and smoke spread quickly, the prompt response of the fire department helped prevent further damage.
According to the Seoul Fire and Disaster Headquarters, 23 fire trucks and 83 firefighters were dispatched immediately after the report was received. 
The firefighters actively worked at the scene to extinguish the fire, and the blaze was fully contained 37 minutes later, at 5:26 PM. 
The scene was marked by the tireless efforts of the firefighting team as they worked to completely put out the flames.
While there were no direct casualties from the fire, the evacuation process caused considerable tension and anxiety among the residents, with some reporting psychological distress afterward. 
Various theories about the cause of the fire were circulating among the residents, but the fire department is currently investigating to determine the exact cause of the incident.
"""

print(summarize(text))
