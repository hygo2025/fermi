"""
Script de Validação e Análise do Canonical Listing ID

Este script ajuda a validar a qualidade dos clusters criados pela
estratégia de Item Canonicalization e fornece insights sobre:
- Redução de esparsidade
- Distribuição de tamanho de clusters
- Impacto no histórico de interações

Uso:
    spark-submit scripts/validate_canonical_id.py
    
    # Ou via Python
    python scripts/validate_canonical_id.py
"""

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from src.utils.enviroment import get_config
from src.utils import log


def analyze_cluster_distribution(mapping_df):
    """
    Analisa a distribuição de tamanho dos clusters canônicos
    
    Args:
        mapping_df: DataFrame com colunas [anonymized_listing_id, canonical_listing_id, listing_id_numeric]
    """
    log("\n" + "="*80)
    log("ANÁLISE DE DISTRIBUIÇÃO DE CLUSTERS")
    log("="*80)
    
    cluster_sizes = (
        mapping_df
        .groupBy("canonical_listing_id")
        .agg(
            F.count("anonymized_listing_id").alias("num_listings_agrupados"),
            F.countDistinct("anonymized_listing_id").alias("distinct_listings")
        )
    )
    
    # Estatísticas descritivas
    stats = cluster_sizes.select("num_listings_agrupados").describe()
    log("\n Estatísticas de Tamanho de Clusters:")
    stats.show()
    
    # Distribuição por faixas
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
    
    # Top 10 maiores clusters
    log("\n Top 10 Maiores Clusters:")
    top_clusters = (
        cluster_sizes
        .orderBy(F.desc("num_listings_agrupados"))
        .limit(10)
    )
    top_clusters.show()
    
    return cluster_sizes


def calculate_sparsity_reduction(mapping_df, events_df=None):
    """
    Calcula a redução de esparsidade proporcionada pelo canonical_id
    
    Args:
        mapping_df: DataFrame de mapeamento
        events_df: DataFrame de eventos (opcional, para análise de interações)
    """
    log("\n" + "="*80)
    log("ANÁLISE DE REDUÇÃO DE ESPARSIDADE")
    log("="*80)
    
    total_original_ids = mapping_df.select("anonymized_listing_id").distinct().count()
    total_canonical_ids = mapping_df.select("canonical_listing_id").distinct().count()
    
    reduction_pct = ((total_original_ids - total_canonical_ids) / total_original_ids) * 100
    
    log(f"\n Redução de IDs:")
    log(f"   - IDs Originais: {total_original_ids:,}")
    log(f"   - IDs Canônicos: {total_canonical_ids:,}")
    log(f"   - Redução: {reduction_pct:.2f}%")
    
    if events_df:
        log("\n Impacto nas Interações:")
        
        # Interações por ID original
        original_interactions = (
            events_df
            .groupBy("anonymized_listing_id")
            .count()
            .select(F.avg("count").alias("avg_interactions_original"))
        )
        
        # Interações por ID canônico
        # Nota: Se events_df não tiver canonical_id, fazer JOIN
        # Se já tiver, usar diretamente
        canonical_interactions = (
            events_df
            .join(mapping_df.select("anonymized_listing_id", "canonical_listing_id"), 
                  on="anonymized_listing_id",
                  how="left")  # LEFT para manter todos eventos
            .groupBy("canonical_listing_id")
            .count()
            .select(F.avg("count").alias("avg_interactions_canonical"))
        )
        
        log("   Média de Interações:")
        original_interactions.show()
        canonical_interactions.show()


def identify_problematic_clusters(mapping_df, listings_df):
    """
    Identifica clusters que podem estar agrupando itens muito diferentes
    
    Args:
        mapping_df: DataFrame de mapeamento
        listings_df: DataFrame processado de listings
    """
    log("\n" + "="*80)
    log("ANÁLISE DE QUALIDADE DOS CLUSTERS")
    log("="*80)
    
    # Juntar mapeamento com features originais
    # Nota: listings_df JÁ tem canonical_listing_id, então usar diretamente
    # sem necessidade de JOIN (evita duplicação)
    enriched = listings_df
    
    # Clusters com alta variação de preço (indicativo de agrupamento incorreto)
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
            "price_cv",  # Coeficiente de variação
            F.col("price_stddev") / F.col("avg_price")
        )
        .filter(F.col("num_listings") > 1)  # Apenas clusters com múltiplos itens
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
    """
    Inspeciona exemplos de clusters para validação manual
    
    Args:
        mapping_df: DataFrame de mapeamento
        listings_df: DataFrame processado de listings
        num_samples: Número de clusters a amostrar
    """
    log("\n" + "="*80)
    log("INSPEÇÃO DETALHADA DE CLUSTERS (AMOSTRA)")
    log("="*80)
    
    # Selecionar clusters de tamanho médio (entre 2 e 10 itens)
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
        
        # Buscar listings desse cluster
        # Filtrar diretamente por canonical_listing_id (já presente em listings_df)
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


def export_validation_report(cluster_stats, output_path):
    """
    Exporta relatório de validação para CSV
    
    Args:
        cluster_stats: DataFrame com estatísticas de clusters
        output_path: Caminho para salvar o relatório
    """
    log(f"\n💾 Exportando relatório para: {output_path}")
    
    cluster_stats.coalesce(1).write.mode("overwrite").csv(
        output_path,
        header=True
    )
    
    log(" Relatório exportado com sucesso!")


def main():
    """Função principal de validação"""
    spark = SparkSession.builder \
        .appName("Canonical ID Validation") \
        .getOrCreate()
    
    config = get_config()
    
    log(" Iniciando Validação do Canonical Listing ID\n")
    
    # Carregar dados
    log("📂 Carregando dados...")
    mapping_df = spark.read.parquet(config['raw_data']['listing_id_mapping_path'])
    listings_df = spark.read.parquet(config['raw_data']['listings_processed_path'])
    
    # Executar análises
    cluster_stats = analyze_cluster_distribution(mapping_df)
    calculate_sparsity_reduction(mapping_df)
    identify_problematic_clusters(mapping_df, listings_df)
    sample_cluster_inspection(mapping_df, listings_df, num_samples=3)
    
    # Exportar relatório
    report_path = config['raw_data'].get('validation_report_path', 'data/reports/canonical_validation')
    export_validation_report(cluster_stats, report_path)
    
    log("\n Validação concluída!")
    
    spark.stop()


if __name__ == "__main__":
    main()
