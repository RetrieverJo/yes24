package org.herring

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Yes24 Project. Preprocessing for raw data.
  * Making "filters" file. Must be run most first.

  * @author hyunje
  * @since 1/14/16, 10:52 AM.
  */
object PreProcessing {
    val categories = Array("인문", "자기계발", "국내문학", "해외문학", "종교")

    def main(args: Array[String]): Unit = {
        //사용될 Context 생성
        val conf = new SparkConf().setAppName("Yes24 PreProcessing")
        val sc = new SparkContext(conf)
        val sqlContext = new SQLContext(sc)

        //CSV 파일 읽어서 파싱.
        //단, 대회 규정상 원본 데이터 파일은 제공할 수 없다.
        val df = sqlContext.read.format("com.databricks.spark.csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load("/yes24/yes24-utf.csv")

        //대회측에서 제안한 카테고리에 해당하는 데이터만 필터링.
        val filters = df.rdd.filter { r =>
            val category = r.getAs[String]("category")
            categories.contains(category)
        }.cache()
        //필터링된 결과 저장.
        filters.coalesce(1).saveAsObjectFile("/yes24/data/filters")

        //책 제목과 ISBN 추출.
        val bookWithId = filters.map { data =>
            val tit = data.getAs[String]("title")
            val isbn = data.getAs[String]("isbn").trim
            (tit, isbn)
        }.distinct().zipWithIndex().map(_.swap).map { data =>
            data._1 + scala.Console.RED + data._2._1 + scala.Console.RED + data._2._2
        }
        //추출한 데이터 저장.
        bookWithId.coalesce(1).saveAsTextFile("/yes24/data/bookWithId")

        //사용자 ID의 인덱스 생성.
        val users = filters.map { data =>
            val user = data.getAs[Int]("uid")
            user
        }.zipWithIndex().map(_.swap).map { data =>
            data._1 + scala.Console.RED + data._2
        }
        //생성한 데이터 저장.
        users.coalesce(1).saveAsTextFile("/yes24/data/uidIndex")
    }
}
