package org.herring

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.Row
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Description
  *
  * @author hyunje
  * @since   2015. 11. 23. 
  */
object LDAUserALS {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Yes24 Comparison").setMaster("spark://Hyunje-macbook:7077")
        val sc = new SparkContext(conf)

        val filters = sc.objectFile[Row]("/yes24/filters")
        val users = sc.textFile("/yes24/uidIndex").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val id = tokens.apply(0)
            val oid = tokens.apply(1)
            (oid, id)
        }

        val items = sc.textFile("/yes24/bookWithId").map { line =>
            val tokens = line.split("\\u001B\\[31m")
            val id = tokens.apply(0)
            val title = tokens.apply(1).trim
            //            val isbn = tokens.applyOrElse(2, "-")
            (title, id)
        }

        val userMap = users.collectAsMap()
        val itemMap = items.collectAsMap()

        val userItem = filters.map { data =>
            val uid = userMap.getOrElse(data.getAs[Int]("uid").toString.trim, "-")
            val iid = itemMap.getOrElse(data.getAs[String]("title").trim, "-")
            (uid, iid)
        }.distinct().map {
            case (uid, iid) => Rating(uid.toInt, iid.toInt, 1.0)
        }
    }
}
