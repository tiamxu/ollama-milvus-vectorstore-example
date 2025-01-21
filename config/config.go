package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	VectorStore VectorStoreConfig `yaml:"vector_store"`
	Ollama      OllamaConfig      `yaml:"ollama"`
	Aliyun      AliyunConfig      `yaml:"aliyun"`
	Processing  ProcessingConfig  `yaml:"processing"`
	MySQL       MySQLConfig       `yaml:"mysql"`
	ModelType   string            `yaml:"model_type"` // ollama or aliyun
}

type VectorStoreConfig struct {
	Type   string       `yaml:"type"`
	Milvus MilvusConfig `yaml:"milvus"`
	Qdrant QdrantConfig `yaml:"qdrant"`
}

type MySQLConfig struct {
	DSN string `yaml:"dsn"`
}

type MilvusConfig struct {
	Address    string      `yaml:"address"`
	DBName     string      `yaml:"db_name"`
	Collection string      `yaml:"collection"`
	Index      IndexConfig `yaml:"index"`
}

type QdrantConfig struct {
	Address    string `yaml:"address"`
	Host       string `yaml:"host"`
	Port       int    `yaml:"port"`
	Collection string `yaml:"collection"`
}

type OllamaConfig struct {
	Address       string  `yaml:"address"`
	LLMModel      string  `yaml:"llm_model"`
	EmbedderModel string  `yaml:"embedder_model"`
	Temperature   float64 `yaml:"temperature"`
}

type AliyunConfig struct {
	BaseURL        string `yaml:"base_url"`
	APIKey         string `yaml:"api_key"`
	LLMModel       string `yaml:"llm_model"`
	EmbeddingModel string `yaml:"embedding_model"`
}

type ProcessingConfig struct {
	ChunkSize      int     `yaml:"chunk_size"`
	ChunkOverlap   int     `yaml:"chunk_overlap"`
	TopK           int     `yaml:"top_k"`
	ScoreThreshold float64 `yaml:"score_threshold"`
}

type IndexConfig struct {
	Type       string `yaml:"type"`
	MetricType string `yaml:"metric_type"`
	NList      int    `yaml:"nlist"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	if err := validateConfig(&cfg); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &cfg, nil
}

func validateConfig(cfg *Config) error {
	if cfg.VectorStore.Milvus.Address == "" {
		return fmt.Errorf("milvus address is required")
	}
	if cfg.ModelType == "ollama" && cfg.Ollama.Address == "" {
		return fmt.Errorf("ollama address is required when using ollama model")
	}
	if cfg.ModelType == "aliyun" && cfg.Aliyun.APIKey == "" {
		return fmt.Errorf("aliyun api key is required when using aliyun model")
	}
	if cfg.ModelType != "ollama" && cfg.ModelType != "aliyun" {
		return fmt.Errorf("model_type must be either 'ollama' or 'aliyun'")
	}
	if cfg.Processing.ChunkSize <= 0 {
		return fmt.Errorf("chunk size must be positive")
	}
	return nil
}

// func (c *MilvusConfig) Validate() error {
// 	if c.Address == "" {
// 		return fmt.Errorf("milvus address is required")
// 	}
// 	if c.Collection == "" {
// 		return fmt.Errorf("milvus collection name is required")
// 	}
// 	return nil
// }

// func (c *QdrantConfig) Validate() error {
// 	if c.Address == "" {
// 		return fmt.Errorf("qdrant address is required")
// 	}
// 	if c.Collection == "" {
// 		return fmt.Errorf("qdrant collection name is required")
// 	}
// 	return nil
// }
