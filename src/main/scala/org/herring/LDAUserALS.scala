package org.herring

import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos
import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}

/**
  * LDA + User-based LDA Clustering + ALS
  *
  * ref: https://gist.github.com/jkbradley/ab8ae22a8282b2c8ce33
  *
  * @author hyunje
  * @since   2015. 11. 23. 
  */
object LDAUserALS {
    val minLength = 2
    val rank = 10
    val numIterations = 10
    val alpha = 0.01
    val lambda = 0.01
    val k: Int = 3


    def main(args: Array[String]) {
        val conf = new SparkConf()
            .setAppName("Yes24 LDA + User Clustering + ALS")
            .setMaster("spark://Hyunje-macbook:7077")
            .set("spark.default.parallelism", "4")

        val sc = new SparkContext(conf)
        sc.setCheckpointDir("/yes24/data/checkpoint")

        //해당 카테고리만 필터링 된 데이터
        val filters = sc.objectFile[Row]("/yes24/data/filters")
        //            .filter(r => r.getAs[String]("category") == "인문")
        //                .filter(r => r.getAs[String]("category") == "자기계발")
        //                .filter(r => r.getAs[String]("category") == "국내문학")
        //                .filter(r => r.getAs[String]("category") == "해외문학")
        //                .filter(r => r.getAs[String]("category") == "종교")

        val users = sc.textFile("/yes24/data/uidIndex").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val nid = tokens.apply(0)
            val oid = tokens.apply(1)
            (oid, nid)
        }
        val userOIdNId = users.collectAsMap()
        val userNIdOId = users.map(_.swap).collectAsMap()

        //아이템 정보 로드
        val items = sc.textFile("/yes24/data/bookWithId").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val id = tokens.apply(0)
            val title = tokens.apply(1).trim
            //            val isbn = tokens.applyOrElse(2, "-")
            (title, id)
        }
        val bookTitleId = items.collectAsMap()
        val bookIdTitle = items.map(_.swap).collectAsMap()

        //책 정보(소개 정보 로드)
        val bookData = sc.textFile("/yes24/data/bookData").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val bookId = tokens.apply(0)
            val title = tokens.apply(1)
            val isbn = tokens.apply(2)
            val yes24id = tokens.apply(3)
            val intro = if (tokens.length == 5) tokens.apply(4) else title
            (bookId, intro)
        }
        val bookDataMap = bookData.collectAsMap()

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

        userNouns.cache()

        //LDA와 클러스터링에 사용될 사용자별 Document 생성
        val documents = userNouns.map { case (id, nouns) =>
            val counts = new scala.collection.mutable.HashMap[Int, Double]()
            nouns.foreach { term =>
                if (bWordMap.value.contains(term)) {
                    val idx = bWordMap.value(term)
                    counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
                }
            }
            (id.toLong, Vectors.sparse(bWordMap.value.size, counts.toSeq))
        }

        documents.cache()

        //LDA 모델 직점 생성하기
        /*
        val numTopics = 20
        val lda = new LDA().setK(numTopics).setMaxIterations(150).setCheckpointInterval(10).setOptimizer("em")
        val preLdaModel = lda.run(documents)
        val modelName = "/yes24/ldamodel/inmoon"
        preLdaModel.save(sc, modelName)
        val ldaModel = DistributedLDAModel.load(sc, modelName)
        */

        //LDA 기존 모델 불러오기
        //        val ldaModel = DistributedLDAModel.load(sc, "/yes24/ldamodel/em-20t-100n")
        //        val ldaModel = DistributedLDAModel.load(sc, "/yes24/ldamodel/em")
        val ldaModel = DistributedLDAModel.load(sc, "/yes24/data/ldamodel/em-20t-200n")

        //LDA 수행 결과 기반의 클러스터링 수행
        val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
        val userTopicDistribution = ldaModel.topTopicsPerDocument(k)
        val clusteringResult = userTopicDistribution.flatMap { case (uid, tids, tweights) =>
            tids.map(t => (uid, t)).zip(tweights).map(t => (t._1._1, t._1._2, t._2))
        }
        val userCluster = clusteringResult.map(i => (i._1, i._2))
        val bUserCluster = sc.broadcast(userCluster.collect())
        userCluster.cache()
        val groupedUserCluster = userCluster.groupByKey()

        println("Num of Topics: " + ldaModel.k)

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

        ratingForEachCluster.cache()

        println(ratingForEachCluster.take(1).head)
        println("Num of Cluster: " + ratingForEachCluster.count())

        //각 클러스터별로 ALS 수행
        val numOfClusters = ratingForEachCluster.count().toInt
        //미리 수행한 추천 결과 불러오기
        val recResult = sc.objectFile[(Int, Int, Array[Rating])]("/yes24/recResult") //(Cluster#, uId, Rec)

        /*
        val recResult = new ArrayBuffer[(Int, Int, Array[Rating])]() //(Cluster#, uId, Rec)

        for (cnum <- 0 until numOfClusters) {
            println("Cluster: "+cnum)
            val ratings = ratingForEachCluster.filter(_._1 == cnum).take(1).head._2
            val ratingsRdd = sc.parallelize(ratings)
            ratingsRdd.cache()

            val model:MatrixFactorizationModel = ALS.trainImplicit(ratingsRdd, rank, numIterations, lambda, alpha)
//            val uid = ratings.head.user
//            val rec = model.recommendProducts(uid, 10)
//            println("rec: "+rec)

            val users = bUserCluster.value.filter(_._2 == cnum).map(_._1.toInt)

            for(uid <- users) {
                val rec = model.recommendProducts(uid, rank)
                recResult += ((cnum, uid, rec))
            }

            ratingsRdd.unpersist()
        }

                val recResultRdd = sc.parallelize(recResult)

*/
        //(user, cluster, ratings)
        val recResultRdd = recResult.map(l => (l._2, l._1, l._3))
        //        recResultRdd.saveAsObjectFile("/yes24/recResult")
        recResultRdd.cache()

        //        val recResultByUser = recResultRdd.groupBy(_._2)
        //        recResultByUser.take(1).foreach(_._2.foreach(l => println("user: "+l._2+", cluster: "+l._1+", count: "+l._3.length)))


        //추천 결과를 이용해 클러스터별 가중치를 이용한 추천 계산
        val userDistSum = userTopicDistribution.map { dist => (dist._1.toInt, dist._3.sum) }.collectAsMap()
        val bUserDistSum = sc.broadcast(userDistSum)
        val recResultTuple = recResultRdd.map(l => ((l._1, l._2), l._3))
        val userItemSim = clusteringResult.map(l => ((l._1.toInt, l._2), l._3)).join(recResultTuple)
        val finalRecommendationResult = userItemSim.flatMap { case ((uid, cid), (dist, ratings)) =>
            ratings.map(r => ((uid, r.product), r.rating * dist))
        }.groupByKey().map { case ((uid, iid), ratings) =>
            val itemSum = ratings.sum
            val distSum = bUserDistSum.value(uid)
            Rating(uid, iid, itemSum / distSum)
        }.groupBy(_.user).map { case (uid, itemRatings) =>
            val sortedItems = itemRatings.toArray.sortBy(-_.rating).take(rank)
            (uid, sortedItems)
        }

        //최종 추천 계산
        val filteredRecommendationResult = finalRecommendationResult.flatMap { case (uid, sortedItems) =>
            val oId = userNIdOId.getOrElse(uid.toString, "-")
            sortedItems.map { case (r) =>
                val title = bookIdTitle.getOrElse(r.product.toString, "-")
                (oId, title, r.rating)
            }
        }

        //추천 결과 저장
        filteredRecommendationResult.map { r =>
            r._1 + Console.RED + r._2 + Console.RED + r._3
        }.coalesce(1).saveAsTextFile("/yes24/final")

        /*
        finalRecommendationResult.take(5).foreach(r =>
            r._2.foreach(i => println("user: " + r._1 + ", item: " + i.product + ", score: " + i.rating))
        )
        */


        documents.unpersist()
        userNouns.unpersist()
        ratingForEachCluster.unpersist()
        recResultRdd.unpersist()
    }
}
