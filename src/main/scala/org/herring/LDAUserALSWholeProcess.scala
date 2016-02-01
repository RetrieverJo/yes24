package org.herring

import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * LDA + User-based LDA Clustering + ALS.
  * 이 형태의 추천 과정을 처음부터 끝까지 수행하는 클래스.
  * 수행하기 전에 PreProcessing 클래스가 수행된 상태여야 하며, 결과가 저장될 폴더는 지워진 상태여야 한다.
  *
  * <pre>
        spark-submit
            --master master_URL
            --class org.herring.LDAUserALSWholeProcess
            --packages com.databricks:spark-csv_2.10:1.3.0
            target/yes24-1.0-allinone.jar
  * </pre>
  *
  * @author hyunje
  * @since 1/14/16, 2:46 PM.
  */
object LDAUserALSWholeProcess {
    val minLength = 2           //명사의 최소 길이
    val rank = 10               //한 사용자당 최대 추천 갯수
    val numRecIterations = 10   //ALS 수행 시 반복 횟수
    val alpha = 0.01            //ALS Hyper-parameter
    val lambda = 0.01           //ALD Hyper-parameter
    val topClusterNum: Int = 3  //한 사용자당 고려할 클러스터 갯수
    val numTopics = 20          //Topic Modeling을 수행할 때 생성할 Topic의 갯수
    val maxLDAIters: Int = 150  //LDA 수행 시 반복 횟수

    //Pre-processing 결과가 저장된 위치
    val filtersPath: String = "/yes24/data/filters"
    val uidIndexPath: String = "/yes24/data/uidIndex"
    val bookWithIdPath: String = "/yes24/data/bookWithId"
    val bookDataPath: String = "/yes24/data/bookData"

    //LDA 수행 결과가 저장될 위치. 이미 존재하는 경로면 수행이 실패한다.
    val ldaModelTargetPath = "/yes24/ldamodel/all"

    //최종 수행 결과가 저장돌 위치. 이미 존재하는 경로면 수행이 실패한다.
    val finalResultPath: String = "/yes24/result/final"

    //구분자. scala.console.RED
    val delimiter: String = "\\u001B\\[31m"

    def main(args: Array[String]) {
        //Spark Context 생성.
        val conf = new SparkConf()
                .setAppName("Yes24 LDA + User Clustering + ALS")
                .set("spark.default.parallelism", "4")
        val sc = new SparkContext(conf)

        //==============================================
        //        PreProcessing 결과를 불러오는 과정
        //==============================================
        println("Load pre-processing result")
        println()

        //(해당 카테고리로 필터링된 결과, 사용자의 ID 정보, 아이템 ID 정보, 책 소개 데이터)
        val (filters, users, items, bookData) = loadPreprocessing(sc)

        val userOIdNId = users.collectAsMap()   //(이전 사용자 ID, 새로운 사용자 ID)
        val userNIdOId = users.map(_.swap).collectAsMap()   //(새로운 사용자 ID, 이전 사용자 ID)
        val bookTitleId = items.collectAsMap()  //(책 제목, 책 ID)
        val bookIdTitle = items.map(_.swap).collectAsMap()  //(책 ID, 책 제목)

        println("Finished loading pre-processing result")
        println()


        //==============================================
        //        LDA를 수행하기 위한 데이터 준비과정
        //==============================================

        println("Preparing LDA")
        println()

        //책 소개 형태소 분석
        val bookStemmed = bookData.map { case (id, intro) =>
            val normalized: CharSequence = TwitterKoreanProcessor.normalize(intro)
            val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
            val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)

            val nouns = stemmed.filter(p => p.pos == KoreanPos.Noun).map(_.text).filter(_.length >= minLength)
            (id, nouns) //(책 Id, Iter[책 소개에 등장한 명사])
        }
        val stemmedMap = bookStemmed.collectAsMap() //(책 Id, Seq[명사])
        val bStemmedMap = sc.broadcast(stemmedMap)

        //LDA에 쓰일 Corpus 생성작업 - 단어별 Wordcount
        val wordCount = bookStemmed.flatMap(data => data._2.map((_, 1))).reduceByKey(_ + _).sortBy(-_._2) //(단어, Count)
        val wordMap = wordCount.map(_._1).zipWithIndex().mapValues(_.toInt).collectAsMap() //(단어, 단어 ID)
        val bWordMap = sc.broadcast(wordMap)

        //사용자가 구매한 책들 추출
        val userItem = filters.map { data =>
            val uid = userOIdNId.getOrElse(data.getAs[Int]("uid").toString.trim, "-")
            val iid = bookTitleId.getOrElse(data.getAs[String]("title").trim, "-")
            (uid, iid)  //(사용자 ID, 구매한 책 ID)
        }.distinct()

        //사용자가 구매한 모든 책의 소개글의 명사를 합한 RDD 생성
        val userNouns = userItem.groupByKey().mapValues { v =>
            val temp: Iterable[Seq[String]] = v.map(bStemmedMap.value.getOrElse(_, Seq[String]()))
            val result = temp.fold(Seq[String]()) { (a, b) => a ++ b }
            result
        }   //(사용자 ID, Seq[명사])

        //LDA와 클러스터링에 사용될 사용자별 Document 생성
        val documents = userNouns.map { case (id, nouns) =>
            val counts = new mutable.HashMap[Int, Double]()
            nouns.foreach { term =>
                if (bWordMap.value.contains(term)) {
                    val idx = bWordMap.value(term)
                    counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
                }
            }
            //Creates a sparse vector using unordered (index, value) pairs.
            (id.toLong, Vectors.sparse(bWordMap.value.size, counts.toSeq))  //(사용자 ID, Vector[각 단어의 Count])
        }

        userNouns.persist(StorageLevel.MEMORY_AND_DISK)
        documents.persist(StorageLevel.MEMORY_AND_DISK)

        println("Finished preparing LDA")
        println()

        //==============================================
        //        LDA를 이용한 클러스터링을 수행하는 과정
        //==============================================

        println("Run LDA")
        println()

        val ldaModel: DistributedLDAModel = runLDA(sc, documents)

        //사용된 Document 데이터 메모리에서 해제
        documents.unpersist()
        userNouns.unpersist()

        val (userTopicDistribution, clusteringResult, userCluster) = ldaResultBasedClustering(ldaModel)

        val bUserCluster = sc.broadcast(userCluster.collect())  //Array[(사용자 ID, Topic ID)]
        userCluster.persist(StorageLevel.MEMORY_AND_DISK)
        val groupedUserCluster = userCluster.groupByKey()   //RDD[(사용자 ID, Iter[Topic ID])]
        println("Finished running LDA")
        println()

        //==============================================
        //            각 클러스터별로 추천을 수행
        //==============================================

        println("Preparing recommendation for each cluster")

        val ratingForEachCluster = prepareRecommendation(userItem, groupedUserCluster)
        ratingForEachCluster.persist(StorageLevel.MEMORY_AND_DISK)

        println("Running recommendation for ech cluster")
        println()

        //각 클러스터별로 ALS 수행
        val recResultRdd = runRecommendation(sc, bUserCluster, ratingForEachCluster)
        recResultRdd.persist(StorageLevel.MEMORY_AND_DISK) //(사용자 ID, Cluster ID, 추천 결과 Array[Rating])

        println("Finished running recommendation")
        println()

        //==============================================
        //            각 사용자별로 최종 추천 계산
        //==============================================

        println("Calculate final recommendation for each user")

        val filteredRecommendationResult = finalRecommendation(
            userTopicDistribution,
            clusteringResult,
            recResultRdd,
            userNIdOId,
            bookIdTitle)

        println("Finished calculating final recommendation for each user")
        println()

        //추천 결과 저장
        println("Saving final recommendation result")
        filteredRecommendationResult.map { r =>
            r._1 + Console.RED + r._2 + Console.RED + r._3
        }.coalesce(1).saveAsTextFile(finalResultPath)

        ratingForEachCluster.unpersist()
        recResultRdd.unpersist()
    }

    def loadPreprocessing(sc: SparkContext)
    : (RDD[Row], RDD[(String, String)], RDD[(String, String)], RDD[(String, String)]) = {

        //해당 카테고리만 필터링 된 결과
        val filters = sc.objectFile[Row](filtersPath)

        //사용자 ID의 인덱싱 결과
        val users = sc.textFile(uidIndexPath).map { line =>
            val tokens = line.split(delimiter)
            val nid = tokens.apply(0)
            val oid = tokens.apply(1)
            (oid, nid)
        }
        //아이템 정보 로드
        val items = sc.textFile(bookWithIdPath).map { line =>
            val tokens = line.split(delimiter)
            val id = tokens.apply(0)
            val title = tokens.apply(1).trim
            //            val isbn = tokens.applyOrElse(2, "-")
            (title, id)
        }

        //책 정보(소개 정보 로드)
        val bookData = sc.textFile(bookDataPath).map { line =>
            val tokens = line.split(delimiter)
            val bookId = tokens.apply(0)
            val title = tokens.apply(1)
            //            val isbn = tokens.apply(2)
            //            val yes24id = tokens.apply(3)
            val intro = if (tokens.length == 5) tokens.apply(4) else title
            (bookId, intro)
        }
        (filters, users, items, bookData)
    }

    def runLDA(sc: SparkContext, documents: RDD[(Long, Vector)]): DistributedLDAModel = {

        //LDA 수행
        val lda = new LDA().setK(numTopics).setMaxIterations(maxLDAIters).setCheckpointInterval(10).setOptimizer("em")
        //일단 수행한 LDA 모델 저장
        val preLdaModel = lda.run(documents)

        preLdaModel.save(sc, ldaModelTargetPath)
        val ldaModel = DistributedLDAModel.load(sc, ldaModelTargetPath)
        ldaModel
    }

    def ldaResultBasedClustering(ldaModel: DistributedLDAModel)
    : (RDD[(Long, Array[Int], Array[Double])], RDD[(Long, Int, Double)], RDD[(Long, Int)]) = {

        //LDA 수행 결과 기반의 클러스터링 수행
        val userTopicDistribution = ldaModel.topTopicsPerDocument(topClusterNum) //RDD[(사용자 ID, Array[Topic ID], Array[Topic과의 연관도]]
        val clusteringResult = userTopicDistribution.flatMap { case (uid, tids, tweights) =>
            tids.map(t => (uid, t)).zip(tweights).map(t => (t._1._1, t._1._2, t._2))
        }   //RDD[(사용자 ID, Topic ID, Topic과의 연관도)]
        val userCluster = clusteringResult.map(i => (i._1, i._2))   //RDD[(사용자 ID, Topic ID)]
        (userTopicDistribution, clusteringResult, userCluster)
    }

    def prepareRecommendation(userItem: RDD[(String, String)],
                              groupedUserCluster: RDD[(Long, Iterable[Int])]): RDD[(Int, Array[Rating])] = {

        //각 클러스터별로 추천을 수행하기 위한 데이터 Filtering
        //ratingForEachCluster: 각 클러스터별로 (클러스터 Id, Array[Ratings])
        val ratingForEachCluster = userItem.map(i => (i._1.toLong, Rating(i._1.toInt, i._2.toInt, 1.0))).groupByKey()
                .join(groupedUserCluster)   //RDD[(사용자 ID, Iter[Rating])] + RDD[(사용자 ID, Iter[Cluster ID])]
                .flatMap { uidRatingCluster =>
                    val uid = uidRatingCluster._1
                    val ratings = uidRatingCluster._2._1.toSeq
                    val clusters = uidRatingCluster._2._2
                    clusters.map(cnum => (cnum, ratings))   //(Cluster ID, Seq[Rating])을 FlatMap 으로 생성
                }.groupByKey()  //(Cluster ID, Iter[Seq[Rating]])
                .mapValues(_.reduce((a, b) => a ++ b))  //(Cluster ID, Seq[Rating])
                .mapValues(_.toArray)   //(Cluster ID, Array[Rating])
        ratingForEachCluster    //각 클러스터에 해당된 사람들이 구매한 아이템을 이용한 Rating 정보
    }

    def runRecommendation(sc: SparkContext,
                          bUserCluster: Broadcast[Array[(Long, Int)]],
                          ratingForEachCluster: RDD[(Int, Array[Rating])]): RDD[(Int, Int, Array[Rating])] = {

        val numOfClusters = ratingForEachCluster.count().toInt
        val recResult = new ArrayBuffer[(Int, Int, Array[Rating])]()

        for (cnum <- 0 until numOfClusters) {
            val ratings = ratingForEachCluster.filter(_._1 == cnum).take(1).head._2 //현재 클러스터에 해당하는 Rating만 추출
            val ratingsRdd = sc.parallelize(ratings)
            ratingsRdd.persist(StorageLevel.MEMORY_AND_DISK)

            //추천 수행
            val model: MatrixFactorizationModel = ALS.trainImplicit(ratingsRdd, rank, numRecIterations, lambda, alpha)
            val users = bUserCluster.value.filter(_._2 == cnum).map(_._1.toInt) //현재 클러스터에 해당하는 사용자 추출

            //각 사용자별로 추천 결과 생성
            for (uid <- users) {
                val rec = model.recommendProducts(uid, rank)    //Array[Rating]
                recResult += ((uid, cnum, rec)) //Array[(사용자 ID, Cluster ID, 추천 결과 Array[Rating])]
            }
            ratingsRdd.unpersist()
        }
        val recResultRdd = sc.parallelize(recResult)
        recResultRdd
    }

    def finalRecommendation(userTopicDistribution: RDD[(Long, Array[Int], Array[Double])],
                            clusteringResult: RDD[(Long, Int, Double)],
                            recResultRdd: RDD[(Int, Int, Array[Rating])],
                            userNIdOId: scala.collection.Map[String, String],
                            bookIdTitle: scala.collection.Map[String, String]): RDD[(String, String, Double)] = {

        //추천 결과와 클러스터별 가중치를 이용한 추천 계산
        val userDistSum = userTopicDistribution.map { dist => (dist._1.toInt, dist._3.sum) }.collectAsMap() //가중평균의 분모, (사용자 ID, 가중치들의 합)
        val recResultTuple = recResultRdd.map(l => ((l._1, l._2), l._3))    //((사용자 ID, Cluster ID), 아이템 Array[Rating])
        //((사용자 ID, Cluster ID), 유사도) JOIN ((사용자 ID, Cluster ID), Array[Rating])
        val userItemSim = clusteringResult.map(l => ((l._1.toInt, l._2), l._3)).join(recResultTuple)
        val finalRecommendationResult = userItemSim.flatMap { case ((uid, cid), (sim, ratings)) =>  //((사용자 ID, Cluster ID), (유사도, 아이템))
            ratings.map(r => ((uid, r.product), r.rating * sim))    //((사용자 ID, 아이템 ID), 아이템에 대한 Rating 추정치 * Cluster와의 유사도))
        }.groupByKey().map { case ((uid, iid), ratings) =>  //(사용자 ID, 아이템 ID)를 Key 로 하여 reduce
            val itemSum = ratings.sum   //아이템에 대한 Rating 추정치 * 유사도의 합
            val distSum = userDistSum(uid)  //모든 유사도의 합
            Rating(uid, iid, itemSum / distSum) //가중평균 계산 후 Rating 결과를 Rating 객체로 Wrapping
        }.groupBy(_.user).map { case (uid, itemRatings) =>  //사용자 별로 추천 받은 아이템들을 reduce
            val sortedItems = itemRatings.toArray.sortBy(-_.rating).take(rank)  //내림차순으로 정렬하여 상위 N개 추출
            (uid, sortedItems)
        }

        //추천된 결과를 Raw Data의 사용자 Id와 다시 매칭
        val filteredRecommendationResult = finalRecommendationResult.flatMap { case (uid, sortedItems) =>
            val oId = userNIdOId.getOrElse(uid.toString, "-")
            sortedItems.map { case (r) =>
                val title = bookIdTitle.getOrElse(r.product.toString, "-")  //지금까지 계산했던 아이템의 ID를 기존의 아이템 ID로 변경
                (oId, title, r.rating)
            }
        }
        filteredRecommendationResult
    }
}
