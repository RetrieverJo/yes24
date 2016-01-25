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
    val minLength = 2
    val rank = 10
    val numRecIterations = 10
    val alpha = 0.01
    val lambda = 0.01
    val topClusterNum: Int = 3
    val numTopics = 20
    val maxLDAIters: Int = 150

    val filtersPath: String = "/yes24/data/filters"
    val uidIndexPath: String = "/yes24/data/uidIndex"
    val bookWithIdPath: String = "/yes24/data/bookWithId"
    val bookDataPath: String = "/yes24/data/bookData"
    val finalResultPath: String = "/yes24/result/final"
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

        val (filters, users, items, bookData) = loadPreprocessing(sc)

        val userOIdNId = users.collectAsMap()
        val userNIdOId = users.map(_.swap).collectAsMap()
        val bookTitleId = items.collectAsMap()
        val bookIdTitle = items.map(_.swap).collectAsMap()

        println("Finished loading pre-processing result")
        println()


        //==============================================
        //        LDA를 수행하기 위한 데이터 준비과정
        //==============================================

        println("Preparing LDA")

        //책 소개 형태소 분석
        val bookStemmed = bookData.map { case (id, intro) =>
            val normalized: CharSequence = TwitterKoreanProcessor.normalize(intro)
            val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
            val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)

            val nouns = stemmed.filter(p => p.pos == KoreanPos.Noun).map(_.text).filter(_.length >= minLength)
            (id, nouns)
        }
        val stemmedMap = bookStemmed.collectAsMap()
        val bStemmedMap = sc.broadcast(stemmedMap)

        //LDA에 쓰일 Corpus 생성작업
        val wordCount = bookStemmed.flatMap(data => data._2.map((_, 1))).reduceByKey(_ + _).sortBy(-_._2)
        //        val bWordCount = sc.broadcast(wordCount)
        val wordArray = wordCount.map(_._1).collect()
        //        val bWordArray = sc.broadcast(wordArray)
        val wordMap = wordCount.map(_._1).zipWithIndex().mapValues(_.toInt).collectAsMap()
        val bWordMap = sc.broadcast(wordMap)

        //사용자가 구매한 책들 추출
        val userItem = filters.map { data =>
            val uid = userOIdNId.getOrElse(data.getAs[Int]("uid").toString.trim, "-")
            val iid = bookTitleId.getOrElse(data.getAs[String]("title").trim, "-")
            (uid, iid)
        }.distinct()

        //사용자가 구매한 책의 소개글을 합한 RDD 생성
        val userNouns = userItem.groupByKey().mapValues { v =>
            val temp: Iterable[Seq[String]] = v.map(bStemmedMap.value.getOrElse(_, Seq[String]()))
            val result = temp.fold(Seq[String]()) { (a, b) => a ++ b }
            result
        }

        //LDA와 클러스터링에 사용될 사용자별 Document 생성
        val documents = userNouns.map { case (id, nouns) =>
            val counts = new mutable.HashMap[Int, Double]()
            nouns.foreach { term =>
                if (bWordMap.value.contains(term)) {
                    val idx = bWordMap.value(term)
                    counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
                }
            }
            (id.toLong, Vectors.sparse(bWordMap.value.size, counts.toSeq))
        }

        userNouns.persist(StorageLevel.MEMORY_AND_DISK)
        documents.persist(StorageLevel.MEMORY_AND_DISK)

        println("Finished preparing LDA")
        println()


        //==============================================
        //        LDA를 이용한 클러스터링을 수행하는 과정
        //==============================================

        println("Run LDA")

        val ldaModel: DistributedLDAModel = runLDA(sc, documents)

        //사용된 Document 데이터 메모리에서 해제
        documents.unpersist()
        userNouns.unpersist()

        val (userTopicDistribution, clusteringResult, userCluster) = ldaResultBasedClustering(ldaModel)

        val bUserCluster = sc.broadcast(userCluster.collect())
        userCluster.persist(StorageLevel.MEMORY_AND_DISK)
        val groupedUserCluster = userCluster.groupByKey()
        println("Finished running LDA")
        println()


        //==============================================
        //            각 클러스터별로 추천을 수행
        //==============================================

        println("Preparing recommendation for each cluster")

        val ratingForEachCluster = prepareRecommendation(userItem, groupedUserCluster)
        ratingForEachCluster.persist(StorageLevel.MEMORY_AND_DISK)

        println("Running recommendation for ech cluster")

        //각 클러스터별로 ALS 수행

        val recResultRdd = runRecommendation(sc, bUserCluster, ratingForEachCluster)
        recResultRdd.persist(StorageLevel.MEMORY_AND_DISK)

        println("Finished running recommendation")
        println()


        //==============================================
        //            각 사용자별로 최종 추천 계산
        //==============================================

        println("각 사용자별 최종 추천 계산")

        val filteredRecommendationResult = finalRecommendation(
            userTopicDistribution,
            clusteringResult,
            recResultRdd,
            userNIdOId,
            bookIdTitle)

        println("각 사용자별 최종 추천 계산 완료")
        println()

        //추천 결과 저장
        println("추천 결과 저장")
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
        val preLdaModel = lda.run(documents)
        //일단 수행한 LDA 모델 저장
        val modelTargetPath = "/yes24/ldamodel/all"
        preLdaModel.save(sc, modelTargetPath)
        val ldaModel = DistributedLDAModel.load(sc, modelTargetPath)
        ldaModel
    }

    def ldaResultBasedClustering(ldaModel: DistributedLDAModel)
    : (RDD[(Long, Array[Int], Array[Double])], RDD[(Long, Int, Double)], RDD[(Long, Int)]) = {

        //LDA 수행 결과 기반의 클러스터링 수행
        val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = topClusterNum)
        val userTopicDistribution = ldaModel.topTopicsPerDocument(topClusterNum)
        val clusteringResult = userTopicDistribution.flatMap { case (uid, tids, tweights) =>
            tids.map(t => (uid, t)).zip(tweights).map(t => (t._1._1, t._1._2, t._2))
        }
        val userCluster = clusteringResult.map(i => (i._1, i._2))
        (userTopicDistribution, clusteringResult, userCluster)
    }

    def prepareRecommendation(userItem: RDD[(String, String)],
                              groupedUserCluster: RDD[(Long, Iterable[Int])]): RDD[(Int, Array[Rating])] = {

        //각 클러스터별로 추천을 수행하기 위한 데이터 Filtering
        // TODO: Join으로 해결하는 것이 효과적인가, 아니면 User-Ratings 의 Map을 구성한 후 Broadcasting 하여 처리하는것이 나은가?
        //ratingForEachCluster: 각 클러스터별로 (클러스터 Id, Array[Ratings])
        val ratingForEachCluster = userItem.map(i => (i._1.toLong, Rating(i._1.toInt, i._2.toInt, 1.0))).groupByKey()
                .join(groupedUserCluster)
                .flatMap { uidRatingCluster =>
                    val uid = uidRatingCluster._1
                    val ratings = uidRatingCluster._2._1.toSeq
                    val clusters = uidRatingCluster._2._2
                    clusters.map(cnum => (cnum, ratings))
                }.groupByKey().mapValues(_.reduce((a, b) => a ++ b)).mapValues(_.toArray)
        ratingForEachCluster
    }

    def runRecommendation(sc: SparkContext,
                          bUserCluster: Broadcast[Array[(Long, Int)]],
                          ratingForEachCluster: RDD[(Int, Array[Rating])]): RDD[(Int, Int, Array[Rating])] = {

        val numOfClusters = ratingForEachCluster.count().toInt
        val recResult = new ArrayBuffer[(Int, Int, Array[Rating])]() //(Cluster#, uId, Rec)

        for (cnum <- 0 until numOfClusters) {
            val ratings = ratingForEachCluster.filter(_._1 == cnum).take(1).head._2
            val ratingsRdd = sc.parallelize(ratings)
            ratingsRdd.persist(StorageLevel.MEMORY_AND_DISK)

            val model: MatrixFactorizationModel = ALS.trainImplicit(ratingsRdd, rank, numRecIterations, lambda, alpha)
            val users = bUserCluster.value.filter(_._2 == cnum).map(_._1.toInt)

            for (uid <- users) {
                val rec = model.recommendProducts(uid, rank)
                recResult += ((cnum, uid, rec))
            }
            ratingsRdd.unpersist()
        }
        val recResultRdd = sc.parallelize(recResult).map(l => (l._2, l._1, l._3))
        recResultRdd
    }

    def finalRecommendation(userTopicDistribution: RDD[(Long, Array[Int], Array[Double])],
                            clusteringResult: RDD[(Long, Int, Double)],
                            recResultRdd: RDD[(Int, Int, Array[Rating])],
                            userNIdOId: scala.collection.Map[String, String],
                            bookIdTitle: scala.collection.Map[String, String]): RDD[(String, String, Double)] = {

        //추천 결과와 클러스터별 가중치를 이용한 추천 계산
        val userDistSum = userTopicDistribution.map { dist => (dist._1.toInt, dist._3.sum) }.collectAsMap()
        val recResultTuple = recResultRdd.map(l => ((l._1, l._2), l._3))
        val userItemSim = clusteringResult.map(l => ((l._1.toInt, l._2), l._3)).join(recResultTuple)
        val finalRecommendationResult = userItemSim.flatMap { case ((uid, cid), (dist, ratings)) =>
            ratings.map(r => ((uid, r.product), r.rating * dist))
        }.groupByKey().map { case ((uid, iid), ratings) =>
            val itemSum = ratings.sum
            val distSum = userDistSum(uid)
            Rating(uid, iid, itemSum / distSum)
        }.groupBy(_.user).map { case (uid, itemRatings) =>
            val sortedItems = itemRatings.toArray.sortBy(-_.rating).take(rank)
            (uid, sortedItems)
        }

        //추천된 결과를 Raw Data의 사용자 Id와 다시 매칭
        val filteredRecommendationResult = finalRecommendationResult.flatMap { case (uid, sortedItems) =>
            val oId = userNIdOId.getOrElse(uid.toString, "-")
            sortedItems.map { case (r) =>
                val title = bookIdTitle.getOrElse(r.product.toString, "-")
                (oId, title, r.rating)
            }
        }
        filteredRecommendationResult
    }
}
