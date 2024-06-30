import os
import pandas as pd
import matplotlib.pyplot as plt
import openai
import requests
from tqdm import tqdm
import time

from google.colab import drive
drive.mount('/journal/drive')

# OpenAI API 키값 설정하기
openai.api_key = "INPUT YOUR API KEY" # "" 안에는 본인 계정 open api key 값을 넣어야 합니다.
GPT_API_URL = "https://api.openai.com/v1/chat/completions"

# 데이터셋 가져오기
df =  pd.read_table('/journal/drive/MyDrive/Colab_Notebooks/bab2min_corpus_master_sentiment_naver_shopping.txt', names=['Rating', 'Review Text'])

# 윗쪽 200개 데이터만 사용
df = df.iloc[0:200]

df['Rating'].value_counts(normalize=True).sort_index()

# 띄어쓰기 기준으로 리뷰 길이 체크
review_list = []

for review in df['Review Text']:
  split= review.split()
  review_list.append(split)

print('리뷰의 최대 단어 수 :', max(len(review) for review in review_list))
print('리뷰의 평균 단어 수 :', sum(map(len, review_list))/len(review_list))
plt.hist([len(review) for review in review_list], bins=50)
plt.xlabel('length of review')
plt.ylabel('number of review')
#plt.show()

## ChatGPT API를 활용한 감정분석

# 리뷰를 분석하기 위한 함수 작성
def analyze_review(review):

  try:
    messages = [
            {"role": "system", "content": "너는 사람들이 쓴 글의 감정을 분석하고 알아내는 AI 언어모델 역할을 해줘."},
            {"role": "user", "content": f"다음 글들은 우리 고객이 쓴 글들이야. 이 글을 쓴 사람들의 감정이 긍정인지 부정인지 판단해서 알려주고, 답은 오직 '긍정' 또는 '부정'  둘 중 하나의 단어로 대답해야해: {review}"}
        ]

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=3,
            n=1,
            stop=None,
            temperature=0
        )

    response= completion.choices[0].message.content
    print(response)
    return response

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return analyze_review(review)

  except openai.error.ServiceUnavailableError as e:
    retry_time = 10  # Adjust the retry time as needed
    print(f"Service is unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return analyze_review(review)

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return analyze_review(review)

# 리뷰 분석해 저장하기
sentiments = []

for review in tqdm(df["Review Text"]):
    sentiment = analyze_review(review)
    sentiments.append(sentiment)

df["Sentiment"] = sentiments

# 엑셀파일로 출력하기
df.to_excel('/journal/drive/MyDrive/Colab_Notebooks/reviews_analyzed_sentiment.xlsx', index=False)
