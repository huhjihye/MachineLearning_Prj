import inline as inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


from sklearn.linear_model import LinearRegression

#데이터셋 불러오기 & 기초 전처리
#데이터셋 가져온 캐글 주소
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
# year_built : 건물이 지어진 연도 - 종속변수
# sale_price : 부동산 판매 가격 - 독립변수
r1 = pd.read_csv('txt/train.csv')
r1.columns = [col.lower() for col in r1.columns]
t1 = r1.query('yearbuilt >= 1980').reset_index(drop=True)



########데이터 분석 목적1 - 이해 & 인과관계 관측을 위한 데이터 분석 (통계 해석학)
# SDEM 방식 사용
# 1. Significant(유의성 확인)
# 2. Direction(방향성 확인)
# 3. Effect Size(효과 크기 측정)
# 4. Model Fit(모델 적합성 확인)

# 반응 변수 정의
y = t1.saleprice

# 예측 변수 정의
x = t1.yearbuilt

# 예측 변수에 상수 추가
x = sm.add_constant(x)

# fit linear regression model
model = sm.OLS(y, x).fit()

# view model summary
print(model.summary())


########데이터 분석 목적2 - 예측을 위한 데이터 분석

## 변수의 선언
x = t1.yearbuilt
y = t1.saleprice

## 모델의 선언
linear_regression = LinearRegression()

## 모델-데이터셋의 학습 진행
linear_regression.fit(x.values.reshape(-1, 1), y)

## 예측 실행
print('2010년 건축 부동산 예측 가격:{}'.format(linear_regression.predict([[2010]])[0])
      , 'X변수 계수: {}'.format(linear_regression.coef_[0])
      , sep='\n')