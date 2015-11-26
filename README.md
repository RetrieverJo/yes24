#MANUAL

0. Maven 기반, Scala 10.4, Spark 1.5.1, Hadoop 2.6.2, Package 수행 시 allinone.jar 생성

1. data에 있는 파일을 HDFS에 업로드

2. spark-shell 을 통해서 package된 파일 실행
 
 
#Algorithm

##I.	주어진 데이터

-	사용자가 책을 구매한 로그
-	구매한 책의 소개글

##II.	핵심 알고리즘

-	Collaborative Filtering(CF)
-	Latent Dirichlet Allocation(LDA)
-	Matrix Factorization-based Recommendation (ALS)

##III.	동작 환경

-	Spark 1.5.x

##IV.	가능한 추천 알고리즘

1.	LDA + CF
2.	LDA + item-based clustering + CF
3.	LDA + user-based clustering + ALS

##V.	비교 대조군

비교를 하는 과정에서 주어진 데이터셋을 9:1로 나누어 9를 트레이닝 셋, 1을 테스트 셋으로 한다.

-	ALS-based recommendation

90%의 데이터로 Spark 기본 추천 알고리즘을 이용해 평가를 수행한다. 트레이닝 셋에 속해있는 각 사용자별 추천 갯수의 총합 대비 각 사용자별로 테스트셋에 있는 데이터와 몇개가 일치하는지 판단한다.

##VI.	관련 알고리즘

###i.	Collaborative Filtering

전통적인 Collaborative Filtering은 아이템 – 아이템 혹은 사용자 – 사용자 사이의 유사도를 계산하여 그것을 이용해 추천을 수행하는 방법을 일컫는다. Item-based collaborative filtering 은 사용자들이 동시에 구매한 아이템들을 이용하여 Matrix를 생성하고, 각 행(아이템)을 Vector로 간주하여 모든 아이템 사이의 유사도를 계산한다. 그리고 계산된 유사도를 내림차순으로 정렬하여 상위 N 개를 해당 아이템과 연관성이 높은 아이템으로 간주하여 추천한다. 평점 형태의 데이터라면 각 아이템 사이의 유사도에 사용자들이 남긴 평점을 곱하여 가중평균을 계산하지만, 사용하고자 하는 데이터는 Binary 형태의 데이터이므로 단순히 가장 유사한 아이템 N 개를 추천해주어도 무방하다.
사용자 기반의 Collaborative filtering은 사용자 – 아이템 Matrix에서 각 행(사용자)을 Vector로 간주하여 모든 사용자 사이의 유사도를 계산한다. 그리고 K-nn 과 같은 알고리즘을 이용하거나 단순히 가장 유사도가 높은 최상위 몇 개의 사용자를 추출하여 각 사용자별로 비슷한 사용자의 그룹을 구한다. 추출된 비슷한 사용자 그룹을 기반으로 하여 가중평균등을 이용해 사용자가 평가를 내리지 않은 아이템에 대해 예상치를 계산하고, 그 값이 높은 순서로 정렬하여 추천을 수행한다.

###ii.	LDA

LDA는 각 Document에 존재하는 단어의 분포를 이용하여 각 Document가 어떠한 Topic을 갖는지, 그 Topic에 대한 분포를 확률적으로 계산하는 알고리즘이다. 하지만 Topic 정보는 사용자가 직접 입력하는 것이 불가능하다. 때문에 LDA를 N개의 Topic으로 클러스터링을 해 주는 것으로 간주하는 의견들이 있고, 실제 Apache Spark 에서는 LDA를 Clustering으로 구분하였다.
LDA의 입력으로는 각 Document 별로 전체 데이터 셋에 존재하는 각 단어의 분포, 즉 Document – Word matrix이 있으며, 몇 개의 Topic을 기준으로 클러스터링을 수행할 것인지에 대한 K 값이 존재한다. LDA를 수행시키게 되면 각 Document별로 어느 Topic과 가까운지에 대한 분포와 각 Topic 별 Word의 분포, 그리고 Topic – Word 분포를 이용한 각 Topic 별 최상위 Word등에 대해 알 수 있다.

###iii.	Alternating Least Squares(ALS)

ALS가 제안된 곳은 Netflix Prize(2009) 이다. Netflix Prize는 Netflix에서 제공한 영화 평점 데이터를 이용하여 추천 알고리즘을 제안하는 대회였는데, ALS는 해당 연도의 대회에서 우승을 한 알고리즘이다. Matrix Factorization을 기반으로 추천 알고리즘이 진행된다.
입력은 사용자 – 아이템 평점 Matrix 인데, SVD의 동작 과정과 비슷하게 K 개의 Factor를 이용해 두 개의 작은 Matrix로 변경한다. 그리고 그것을 다시 결합하여 기존의 평점 Matrix와 어느정도 오차가 발생하는지 계산하고, 그 오차를 줄이는 방향으로 둘로 나뉘어진 Matrix를 최적화 시켜나간다. 특정 반복 횟수 또는 일정 오차 이내로 오차가 줄어들면 반복을 종료한다. 사용자가 평점을 매기지 않은 아이템에 대해서는 분할된 두 개의 Matrix의 행과 열을 곱하여 계산한다.

##VII.	알고리즘 명세

####LDA + Item-based CF

이 방법은 기존에 단순히 아이템 – 아이템 사이의 Vector 혹은 사용자 – 아이템 사이의 Vector 사이의 유사도를 계산하던 것에서 LDA를 이용하여 새로운 Vector를 생성하는 것이다.

1.	각 책 별로 소개글에 등장하는 단어를 이용하여 각 책을 Document, 소개글의 단어를 Word로 하는 LDA를 수행한다.
2.	Document(책) – Topic 분포의 각 행을 Item(책)의 Vector로 간주한다.
3.	각 Vector 사이의 유사도를 계산한다.
4.	계산된 유사도를 기반으로 각 아이템별 Top-N 유사 아이템을 추출한다.
5.	해당 아이템을 추천한다.

####LDA + User-based CF

이 방법은 기존에 단순히 아이템 – 아이템 사이의 Vector 혹은 사용자 – 아이템 사이의 Vector 사이의 유사도를 계산하던 것에서 LDA를 이용하여 새로운 Vector를 생성하는 것이다.

1.	각 사용자별로 구매한 책들의 소개글을 합하여 하나의 Document라 가정한 후 각 사용자를 하나의 Document, 각 단어를 Word로 하는 LDA를 수행한다.
2.	Document(사용자) – Topic분포의 각 행을 사용자의 Vector로 간주한다.
3.	각 Vector 사이의 유사도를 계산한다.
4.	계산된 유사도를 기반으로 각 사용자별 Top-N 유사 사용자를 추출한다.
5.	구매하지 않은 아이템에 대해 가중평균을 계산한다.
6.	계산된 가중평가 순으로 아이템을 추천한다.

####LDA + item-based clustering + CF

이 방법은 LDA를 이용하여 아이템을 클러스터링하고, 각각의 클러스터별로 CF를 적용하여 각 클러스터별로 추천된 결과를 종합하여 최종 결과를 도출하는 방법이다. 각 아이템에 대한 LDA의 입력은 LDA + CF의 Item-based 방식과 동일하다. 하지만 Document – Topic의 분포에서 각 Document 별로 가장 높은 분포를 갖는 Topic 이 해당 Document가 속하는 클러스터라고 가정할 수 있다. 그리고 각 Topic 별 확률값의 합은 1이므로 각 확률값을 그대로 가중평균의 가중치로 활용할 수 있다.

1.	각 책 별로 소개글에 등장하는 단어를 이용하여 각 책을 Document, 소개글의 단어를 Word로 하는 LDA를 수행한다.
2.	각 Document 별로 높은 상위 2 개의 Topic에 해당 아이템을 할당한다.
3.	각 클러스터(Topic) 별로 Item – User Matrix를 생성하는데, 이때의 Matrix의 Column(Item)을 그 클러스터에 속하는 Item으로만 구성한다.
4.	각 클러스터의 Item - User Matrix를 이용하여 Item-Item Matrix를 생성한다. 이 때, 각 행과 열에 대한 값은 행과 열의 아이템을 동시에 구매한 것이 Item – User Matrix에 존재하면 1, 그렇지 않으면 0으로 설정한다.
5.	각 행을 하나의 Vector로 하여 다른 Vector와의 유사도를 계산한다.
6.	각 클러스터의 아이템별로 가장 유사한 아이템 N 개를 추출한다.
7.	모든 아이템에 대해 그 아이템이 속하는 클러스터(각 2개)의 유사도와 각 클러스터별로 존재하는 N 개의 유사 아이템의 유사도를 가중평균을 계산하고, 이를  내림차순으로 정렬하여 저장한다.
8.	추천을 수행하고자 하는 사용자가 구매한 아이템들에 대해 과정 (7)에서 저장한 테이블에서 결과를 가지고오고, 같은 아이템의 존재한다면 추가 가중치를 부여한다.
9.	최종 결과를 정렬하여 추천한다.

####LDA + user-based clustering + ALS

이 방법은 LDA를 이용하여 사용자를 클러스터링하고, 각각의 클러스터별로 ALS를 적용하여 적용하여 각 클러스터별로 추천된 결과를 종합하여 최종 결과를 도출하는 방법이다. 각 아이템에 대한 LDA의 입력은 LDA + CF의 User-based 방식과 동일하다. 하지만 Document – Topic의 분포에서 각 Document 별로 가장 높은 분포를 갖는 Topic 이 해당 Document가 속하는 클러스터라고 가정할 수 있다. 그리고 각 Topic 별 확률값의 합은 1이므로 각 확률값을 그대로 가중평균의 가중치로 활용할 수 있다.

1.	각 사용자별로 구매한 책들의 소개글을 합하여 하나의 Document라 가정한 후, 각 사용자를 하나의 Document, 각 단어를 Word로 하는 LDA를 수행한다.
2.	각 Document(사용자) 별로 높은 상위 3 개의 Topic에 해당 아이템을 할당한다.
3.	각 클러스터(Topic)별로 User-Item Matrix를 생성한다. 이 Matrix를 생성할 때, 해당 클러스터에 속하는 사용자만을 이용하여 구성한다.
4.	각 클러스터별로 생성한 User-Item matrix를 이용하여 ALS를 수행한다. ALS의 수행 결과로 각 사용자별 아이템에 대한 예상 수치를 얻을 수 있다.
5.	모든 사용자별로 각 사용자가 속하는 3 개의 클러스터에서 각 클러스터별 유사도와, 각 클러스터의 ALS에서 추천된 N개의 아이템의 아이템과 그 예상 수치를 이용하여 각 아이템별로 가중평균을 구한다. 같은 아이템이 양쪽 클러스터에서 같이 등장한다면 추가 가중치를 부여한다.
6.	계산된 가중평균을 기준으로 내림차순 정렬하여 최대 N 개를 추천한다.

VIII.	수행 결과
IX.	 알고리즘 최적화
1.	Parameter 최적화
2.	비슷한 책 그룹화
3.	유사도 측정방식 변경
4.	
X.	최종 알고리즘
