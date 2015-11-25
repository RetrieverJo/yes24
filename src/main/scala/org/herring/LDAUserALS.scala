package org.herring

import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos
import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.linalg.Vectors
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

    def main(args: Array[String]) {
        val conf = new SparkConf()
            .setAppName("Yes24 LDA + User Clustering + ALS")
            .setMaster("spark://Hyunje-macbook:7077")
            .set("spark.default.parallelism", "4")

        val sc = new SparkContext(conf)

        //해당 카테고리만 필터링 된 데이터
        val filters = sc.objectFile[Row]("/yes24/filters")
        val users = sc.textFile("/yes24/uidIndex").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val id = tokens.apply(0)
            val oid = tokens.apply(1)
            (oid, id)
        }
        val userMap = users.collectAsMap()

        //아이템 정보 로드
        val items = sc.textFile("/yes24/bookWithId").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val id = tokens.apply(0)
            val title = tokens.apply(1).trim
            //            val isbn = tokens.applyOrElse(2, "-")
            (title, id)
        }
        val itemMap = items.collectAsMap()

        //책 정보(소개 정보 로드)
        val bookData = sc.textFile("/yes24/bookData").map { line =>
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
        val bWordCount = sc.broadcast(wordCount)
        val wordArray = bWordCount.value.map(_._1).collect()
        val bWordArray = sc.broadcast(wordArray)
        val wordMap = bWordCount.value.map(_._1).zipWithIndex().mapValues(_.toInt).collectAsMap()
        val bWordMap = sc.broadcast(wordMap)

        //사용자가 구매한 책들 추출
        val userItem = filters.map { data =>
            val uid = userMap.getOrElse(data.getAs[Int]("uid").toString.trim, "-")
            val iid = itemMap.getOrElse(data.getAs[String]("title").trim, "-")
            (uid, iid)
        }.distinct()

        //사용자가 구매한 책의 소개글을 합한 RDD 생성
        val userNouns = userItem.groupByKey().mapValues { v =>
            val temp:Iterable[Seq[String]] = v.map(bStemmedMap.value.getOrElse(_, Seq[String]()))
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

        //LDA 모델 불러오기
        val ldaModel = DistributedLDAModel.load(sc, "/yes24/ldamodel/em-20t-100n")

        /*
        //LDA Training
        val numTopics = 10
        val lda = new LDA().setK(numTopics).setMaxIterations(10).setOptimizer("online")
        val ldaModel = lda.run(documents)
        */

        // Print topics, showing top-weighted 10 terms for each topic.
        val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
        topicIndices.foreach { case (terms, termWeights) =>
            println("TOPIC:")
            terms.zip(termWeights).foreach { case (term, weight) =>
                println(s"${wordArray(term.toInt)}\t$weight")
            }
            println()
        }

        documents.unpersist()
        userNouns.unpersist()
    }
}
