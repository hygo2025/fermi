import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from src.utils.enviroment import get_config
from src.utils import log, make_spark


def analyze_cluster_distribution(mapping_df):
    log("\n" + "-"*80)
    log("CLUSTER DISTRIBUTION ANALYSIS")
    log("-"*80)

    cluster_sizes = (
        mapping_df
        .groupBy("canonical_listing_id")
        .agg(
            F.count("anonymized_listing_id").alias("num_listings_agrupados"),
            F.countDistinct("anonymized_listing_id").alias("distinct_listings")
        )
    )

    stats = cluster_sizes.select("num_listings_agrupados").describe()
    log("\n Cluster Size Statistics:")
    stats.show()


    distribution = (
        cluster_sizes
        .withColumn(
            "cluster_size_range",
            F.when(F.col("num_listings_agrupados") == 1, "1 (Singleton)")
            .when(F.col("num_listings_agrupados") <= 5, "2-5 (Pequeno)")
            .when(F.col("num_listings_agrupados") <= 10, "6-10 (Médio)")
            .when(F.col("num_listings_agrupados") <= 50, "11-50 (Grande)")
            .otherwise("50+ (Muito Grande)")
        )
        .groupBy("cluster_size_range")
        .count()
        .orderBy("cluster_size_range")
    )

    log("\n Distribuição por Faixas de Tamanho:")
    distribution.show()


    log("\n Top 10 Maiores Clusters:")
    top_clusters = (
        cluster_sizes
        .orderBy(F.desc("num_listings_agrupados"))
        .limit(10)
    )
    top_clusters.show()

    return cluster_sizes


def calculate_sparsity_reduction(mapping_df, events_df=None):

    log("\n" + "-"*80)
    log("ANÁLISE DE REDUÇÃO DE ESPARSIDADE")
    log("-"*80)

    total_original_ids = mapping_df.select("anonymized_listing_id").distinct().count()
    total_canonical_ids = mapping_df.select("canonical_listing_id").distinct().count()

    reduction_pct = ((total_original_ids - total_canonical_ids) / total_original_ids) * 100

    log(f"\n Redução de IDs:")
    log(f"   - IDs Originais: {total_original_ids:,}")
    log(f"   - IDs Canônicos: {total_canonical_ids:,}")
    log(f"   - Redução: {reduction_pct:.2f}%")

    if events_df:
        log("\n Impacto nas Interações:")


        original_interactions = (
            events_df
            .groupBy("anonymized_listing_id")
            .count()
            .select(F.avg("count").alias("avg_interactions_original"))
        )

        canonical_interactions = (
            events_df
            .join(mapping_df.select("anonymized_listing_id", "canonical_listing_id"),
                  on="anonymized_listing_id",
                  how="left")
            .groupBy("canonical_listing_id")
            .count()
            .select(F.avg("count").alias("avg_interactions_canonical"))
        )

        log("   Média de Interações:")
        original_interactions.show()
        canonical_interactions.show()


def identify_problematic_clusters(mapping_df, listings_df):

    log("\n" + "-"*80)
    log("ANÁLISE DE QUALIDADE DOS CLUSTERS")
    log("-"*80)




    enriched = listings_df


    price_variance = (
        enriched
        .groupBy("canonical_listing_id")
        .agg(
            F.count("*").alias("num_listings"),
            F.stddev("price").alias("price_stddev"),
            F.avg("price").alias("avg_price"),
            F.min("price").alias("min_price"),
            F.max("price").alias("max_price")
        )
        .withColumn(
            "price_cv",
            F.col("price_stddev") / F.col("avg_price")
        )
        .filter(F.col("num_listings") > 1)
    )

    log("\n️  Clusters com Alta Variação de Preço (CV > 0.3):")
    suspicious_clusters = (
        price_variance
        .filter(F.col("price_cv") > 0.3)
        .orderBy(F.desc("price_cv"))
        .limit(10)
    )
    suspicious_clusters.show()

    log("\n Clusters com Baixa Variação de Preço (CV < 0.1):")
    good_clusters = (
        price_variance
        .filter(F.col("price_cv") < 0.1)
        .orderBy("price_cv")
        .limit(10)
    )
    good_clusters.show()


def sample_cluster_inspection(mapping_df, listings_df, num_samples=5):

    log("\n" + "-"*80)
    log("INSPEÇÃO DETALHADA DE CLUSTERS (AMOSTRA)")
    log("-"*80)


    cluster_sizes = (
        mapping_df
        .groupBy("canonical_listing_id")
        .count()
        .filter((F.col("count") >= 2) & (F.col("count") <= 10))
    )

    sample_clusters = cluster_sizes.sample(False, 0.1).limit(num_samples)

    for row in sample_clusters.collect():
        canonical_id = row["canonical_listing_id"]
        size = row["count"]

        log(f"\n{'─'*80}")
        log(f"Cluster ID: {canonical_id} | Tamanho: {size}")
        log(f"{'─'*80}")



        cluster_listings = (
            listings_df
            .filter(F.col("canonical_listing_id") == canonical_id)
            .select(
                "anonymized_listing_id",
                "lat_region", "lon_region", "zip_code",
                "usable_areas", "bedrooms", "suites", "unit_type",
                "price", "created_at"
            )
        )

        cluster_listings.show(truncate=False)


def main():

    spark = make_spark()

    config = get_config()

    log(" Iniciando Validação do Canonical Listing ID\n")


    log(" Carregando dados...")
    mapping_df = spark.read.parquet(config['raw_data']['listing_id_mapping_path'])
    listings_df = spark.read.parquet(config['raw_data']['listings_processed_path'])


    cluster_stats = analyze_cluster_distribution(mapping_df)
    calculate_sparsity_reduction(mapping_df)
    identify_problematic_clusters(mapping_df, listings_df)
    sample_cluster_inspection(mapping_df, listings_df, num_samples=3)


    log("\n Validação concluída!")

    spark.stop()


if __name__ == "__main__":
    main()
