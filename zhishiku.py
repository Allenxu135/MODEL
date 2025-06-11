import os
import re
import json
import logging
from typing import Iterator, List, Dict, Optional
from dataclasses import dataclass
import multiprocessing as mp

# Big data processing libraries
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, StructType, StructField

# NLP processing libraries
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import ftfy  # Fix text encoding issues

# Deep learning tools
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BertTokenizerFast,
    GPT2TokenizerFast
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration parameters"""
    input_path: str  # Input data path (supports wildcards like /data/*.jsonl)
    output_dir: str  # Output directory
    tokenizer_name: str = "gpt2"  # Pretrained tokenizer name or path
    max_seq_length: int = 1024  # Maximum sequence length
    min_text_length: int = 50  # Minimum text length (filter short texts)
    dedupe_threshold: float = 0.9  # Deduplication similarity threshold
    text_quality_threshold: float = 0.7  # Quality score threshold
    train_ratio: float = 0.95  # Training set ratio
    output_format: str = "tfrecord"  # Output format (tfrecord|hdf5|jsonl)
    num_partitions: int = 100  # Number of output partitions (controls file size)
    language: str = "english"  # Primary language


class BigDataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.spark = self._initialize_spark_session()
        self.tokenizer = self._load_tokenizer()
        self._prepare_output_directory()

    def _initialize_spark_session(self):
        """Initialize Spark session"""
        return SparkSession.builder \
            .appName("LLM Data Preprocessing") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "16g") \
            .config("spark.sql.shuffle.partitions", str(self.config.num_partitions)) \
            .getOrCreate()

    def _load_tokenizer(self):
        """Load or train tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                use_fast=True
            )
            logger.info(f"Loaded tokenizer: {self.config.tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, training new one...")
            tokenizer = self._train_tokenizer()
        return tokenizer

    def _train_tokenizer(self):
        """Train new tokenizer on data subset"""
        # Implementation needs customization based on requirements
        raise NotImplementedError("Custom tokenizer training not implemented")

    def _prepare_output_directory(self):
        """Prepare output directory structure"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "val"), exist_ok=True)

    def _load_data(self):
        """Load raw data into Spark DataFrame"""
        if self.config.input_path.endswith(".jsonl"):
            df = self.spark.read.json(self.config.input_path)
        elif self.config.input_path.endswith(".csv"):
            df = self.spark.read.csv(self.config.input_path, header=True)
        elif self.config.input_path.endswith(".parquet"):
            df = self.spark.read.parquet(self.config.input_path)
        else:  # Assume plain text
            schema = StructType([StructField("text", StringType(), True)])
            df = self.spark.read.text(self.config.input_path, schema=schema)
        return df

    @staticmethod
    def clean_text(text: str) -> str:
        """Text cleaning function"""
        if not text or not isinstance(text, str):
            return ""

        # Fix encoding issues
        text = ftfy.fix_text(text)

        # Remove HTML/XML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Additional custom cleaning rules
        text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
        text = re.sub(r"\b\d{10,}\b", "", text)  # Remove long numbers

        return text

    @staticmethod
    def calculate_text_quality(text: str) -> float:
        """Heuristic text quality scoring (0-1)"""
        if len(text) < 50:
            return 0.0

        # Calculate average sentence length
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0

        avg_len = sum(len(s) for s in sentences) / len(sentences)

        # Calculate symbol ratio
        symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)

        # Composite score (adjust weights based on domain)
        score = 0.3 * min(avg_len / 50, 1.0) + \
                0.4 * (1 - min(max(symbol_ratio - 0.2, 0), 0.5)) + \
                0.3 * min(len(sentences) / 20, 1.0)

        return score

    def _preprocess_data(self, df):
        """Distributed data preprocessing pipeline"""
        # Register UDFs for Spark
        clean_text_udf = F.udf(self.clean_text, StringType())
        quality_score_udf = F.udf(self.calculate_text_quality, StringType())

        # Apply cleaning and quality filtering
        processed_df = df.withColumn("cleaned_text", clean_text_udf(F.col("text"))) \
            .withColumn("quality_score", quality_score_udf(F.col("cleaned_text"))) \
            .filter(F.length(F.col("cleaned_text")) >= self.config.min_text_length) \
            .filter(F.col("quality_score") >= self.config.text_quality_threshold)

        # Deduplication (using MinHash or SimHash for production)
        processed_df = processed_df.dropDuplicates(["cleaned